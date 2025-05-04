import dataclasses
import datetime
import functools
import math
import re
from typing import Optional

import cartopy.crs as ccrs
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
from IPython.display import HTML
import ipywidgets as widgets
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
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

# prerequisits
model_type = {
   "original" : 'GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz',
   "operational" : 'GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz',
   "small" : 'GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz'
}

pretrained_model = model_type[parser.parse_args().model]

with open(f"/geodata2/S2S/DL/GC_input/params/{pretrained_model}", "rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)

    
params = ckpt.params  # 로드된 체크포인트에서 파라미터 가져오기
state = {}  # 초기 상태는 빈 딕셔너리로 설정

# 체크포인트에서 모델 구성 가져오기
model_config = ckpt.model_config
# 체크포인트에서 작업 구성 가져오기
task_config = ckpt.task_config

# 모델 설명 출력
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
diffs_stddev_by_level = xarray.open_dataset("/geodata2/S2S/DL/GC_input/stat/stats_diffs_stddev_by_level.nc")
mean_by_level = xarray.open_dataset("/geodata2/S2S/DL/GC_input/stat/stats_mean_by_level.nc")
stddev_by_level = xarray.open_dataset("/geodata2/S2S/DL/GC_input/stat/stats_stddev_by_level.nc")


eval_steps = parser.parse_args().eval_steps

target_template = his_utils.create_target_dataset(time_steps=eval_steps, 
                   resolution=model_config.resolution, 
                   pressure_levels=len(task_config.pressure_levels))

# start_time: dataset.datetime[0]
forcings = his_utils.create_forcing_dataset(time_steps=eval_steps,
                                              resolution=model_config.resolution,
                                              start_time=dataset.datetime.values[0, 0])
print("3 ================================")

def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):
  """GraphCast 예측기를 구성하고 래핑합니다."""
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
    """구성된 예측기를 사용하여 순전파를 실행합니다."""
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)

@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
    """구성된 예측기를 사용하여 손실과 진단 정보를 계산합니다."""
    predictor = construct_wrapped_graphcast(model_config, task_config)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return xarray_tree.map_structure(
        lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
        (loss, diagnostics)
    )

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
  """매개변수에 대한 손실 함수의 그래디언트를 계산합니다."""
  def _aux(params, state, i, t, f):
    (loss, diagnostics), next_state = loss_fn.apply(
        params, state, jax.random.PRNGKey(0), model_config, task_config,
        i, t, f)
    return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True)(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads

def with_configs(fn):
  """모델 및 작업 구성을 적용하는 유틸리티 함수입니다."""
  return functools.partial(
      fn, model_config=model_config, task_config=task_config)

def with_params(fn):
  """함수가 항상 매개변수와 상태를 받도록 보장합니다."""
  return functools.partial(fn, params=params, state=state)

def drop_state(fn):
  """예측만 반환합니다. 롤아웃 코드에 필요합니다."""
  return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))

if params is None:
    # 매개변수와 상태가 아직 설정되지 않은 경우 초기화합니다.
    params, state = init_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=train_inputs,
        targets_template=train_targets,
        forcings=train_forcings
    )

# 필요한 구성과 매개변수로 함수들을 JIT 컴파일합니다.
loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

# Ensure that the model resolution matches the data resolution
assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
    "Model resolution doesn't match the data resolution. You likely want to "
    "re-filter the dataset list, and download the correct data.")
print("4 ================================")

# Perform autoregressive rollout to generate predictions
predictions = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=target_template,
    forcings=forcings
)

# Display the predictions
predictions.squeeze()[[
   "u_component_of_wind",
   "v_component_of_wind"
    ]].rename({
      "u_component_of_wind":"u",
      "v_component_of_wind":"v"
    }).to_netcdf(parser.parse_args().output)

print("5 ================================")