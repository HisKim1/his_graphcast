import dataclasses
import functools
from typing import Optional

from google.cloud import storage
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
import numpy as np
import xarray
import his_utils
import argparse

parser = argparse.ArgumentParser(description='run GraphCast. LEGGO')
parser.add_argument('--model', type=str, choices=["original", "operational", "small"], required=True)
parser.add_argument('--eval_steps', type=int, required=True)
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)

def parse_file_parts(file_name):
    return dict(part.split("-", 1) for part in file_name.split("_"))

gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")

# Prerequisites
model_type = {
    "original": 'GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz',
    "operational": 'GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz',
    "small": 'GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz'
}

# Load the selected model
pretrained_model = model_type[parser.parse_args().model]

with gcs_bucket.blob(f"params/{pretrained_model}").open("rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)

params = ckpt.params
state = {}  # Initial state

# Get model and task configurations from the checkpoint
model_config = ckpt.model_config
task_config = ckpt.task_config

print("Model description:\n", ckpt.description, "\n")
print("Model config:\n", model_config, "\n")
print("Task config:\n", task_config)
print("1 ================================")

file_name = parser.parse_args().input
dataset = xarray.open_dataset(file_name)

eval_inputs, _, _ = data_utils.extract_inputs_targets_forcings(
    dataset,
    target_lead_times=slice("6h", "0h"),
    **dataclasses.asdict(task_config)
)

print("2 ================================")
diffs_stddev_by_level = xarray.open_dataset("/home/hiskim1/graphcast/testdata/stats/stats_diffs_stddev_by_level.nc")
mean_by_level = xarray.open_dataset("/home/hiskim1/graphcast/testdata/stats/stats_mean_by_level.nc")
stddev_by_level = xarray.open_dataset("/home/hiskim1/graphcast/testdata/stats/stats_stddev_by_level.nc")

eval_steps = parser.parse_args().eval_steps

target_template = his_utils.create_target_dataset(
    time_steps=eval_steps,
    resolution=model_config.resolution,
    pressure_levels=len(task_config.pressure_levels)
)

# Start time: dataset.datetime[0]
forcings = his_utils.create_forcing_dataset(
    time_steps=eval_steps,
    resolution=model_config.resolution,
    start_time=dataset.datetime.values[0, 0]
)
print("3 ================================")

def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):
    predictor = graphcast.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level)
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor

@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)

@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return xarray_tree.map_structure(
        lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
        (loss, diagnostics)
    )

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
    def _aux(params, state, i, t, f):
        (loss, diagnostics), next_state = loss_fn.apply(
            params, state, jax.random.PRNGKey(0), model_config, task_config,
            i, t, f)
        return loss, (diagnostics, next_state)
    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
        _aux, has_aux=True)(params, state, inputs, targets, forcings)
    return loss, diagnostics, next_state, grads

def with_configs(fn):
    return functools.partial(
        fn, model_config=model_config, task_config=task_config)

def with_params(fn):
    return functools.partial(fn, params=params, state=state)

def drop_state(fn):
    return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))

if params is None:
    # Initialize params and state if not set
    params, state = init_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=train_inputs,
        targets_template=train_targets,
        forcings=train_forcings
    )

# JIT compile functions with configurations and parameters
loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

# Ensure that the model resolution matches the data resolution
assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
    "Model resolution doesn't match the data resolution. You likely want to "
    "re-filter the dataset list, and download the correct data.")
print("4 ================================")
print("FGSM Attack")

def add_structured_perturbation(inputs, variables, scale, pattern='gaussian'):
    # Create a copy of the inputs
    perturbed_inputs = inputs.copy(deep=True)
    
    for var in variables:
        data = perturbed_inputs[var].data  # Original data
        
        # Generate perturbation pattern
        if pattern == 'gaussian':
            # Create a Gaussian perturbation
            key = jax.random.PRNGKey(0)
            noise = jax.random.normal(key, shape=data.shape) * scale
        elif pattern == 'uniform':
            # Create a uniform perturbation
            key = jax.random.PRNGKey(0)
            noise = jax.random.uniform(key, shape=data.shape, minval=-scale, maxval=scale)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        # Add perturbation to the data
        perturbed_data = data + noise
        
        # Update the variable in the perturbed inputs
        perturbed_inputs[var].data = perturbed_data
    
    return perturbed_inputs


def fgsm_attack(params, state, model_config, task_config, inputs, targets_template, forcings, epsilon, variables):
    # Extract the data arrays for the variables we want to perturb as JAX arrays
    input_data = {var: jnp.array(inputs[var].data) for var in variables}

    # Flatten the input data to a list of arrays
    input_leaves, input_treedef = tree_flatten(input_data)

    # Define a function that computes the loss given the flattened inputs
    def loss_fn_for_inputs(input_leaves):
        # Reconstruct input_data from leaves
        input_data_reconstructed = tree_unflatten(input_treedef, input_leaves)

        # Create a copy of inputs with the reconstructed data
        perturbed_inputs = inputs.copy(deep=True)
        for var in variables:
            # Assign the reconstructed data to the copy of inputs
            perturbed_inputs[var].data = input_data_reconstructed[var]

        # Compute loss using perturbed inputs
        loss, _ = loss_fn.apply(
            params, state, jax.random.PRNGKey(0), model_config, task_config,
            perturbed_inputs, targets_template, forcings)

        # Extract scalar loss value (mean over all elements)
        loss_value = xarray_jax.unwrap_data(loss.mean(), require_jax=True)
        return loss_value

    # Compute gradients of the loss with respect to the flattened input data
    grad_fn = jax.grad(loss_fn_for_inputs)
    grads_leaves = grad_fn(input_leaves)

    # Apply FGSM perturbation
    perturbed_leaves = []
    for leaf, grad in zip(input_leaves, grads_leaves):
        perturbation = epsilon * jnp.sign(grad)
        perturbed_leaf = leaf + perturbation
        perturbed_leaves.append(perturbed_leaf)

    # Reconstruct the perturbed input data
    perturbed_input_data = tree_unflatten(input_treedef, perturbed_leaves)

    # Create a new inputs dataset with the perturbed data
    perturbed_inputs = inputs.copy(deep=True)
    for var in variables:
        perturbed_inputs[var].data = perturbed_input_data[var]

    return perturbed_inputs

# Variables to perturb
variables = ['2m_temperature']

# Epsilon value for FGSM
epsilon = 0.01


variables = ['2m_temperature']
scale = 0.01  # Adjust the scale as needed
perturbed_inputs = add_structured_perturbation(eval_inputs, 
                                               variables, 
                                               scale, 
                                               pattern='gaussian')


# perturbed_inputs = fgsm_attack(
#     params, state, model_config, task_config,
#     inputs=eval_inputs,  # Original inputs
#     targets_template=target_template,
#     forcings=forcings,
#     epsilon=epsilon,
#     variables=variables
# )

print("5 ================================")

# Perform autoregressive rollout to generate predictions
predictions = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=perturbed_inputs,
    targets_template=target_template,
    forcings=forcings
)

# Save the predictions
predictions.to_netcdf(parser.parse_args().output)
