import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import List
import xarray as xr

class WeatherErrorAccumulationMetricJAX:
    def __init__(
        self,
        cts_data: xr.Dataset,
        perturb_datasets: List[xr.Dataset],
        target_var: str,
        n_samples: int = None,
        use_normal_approx: bool = True
    ):
        """
        Initialize the error accumulation metric calculator.

        Parameters:
        - cts_data: xarray.Dataset containing CTS (continuous-time forecasting) data.
        - perturb_datasets: List of xarray.Datasets containing perturbed data from the generative model.
        - target_var: The target variable (e.g., 'geopotential') to compute the error on.
        - n_samples: Number of perturbation samples to use (default uses all).
        - use_normal_approx: Whether to use normal approximation for KL divergence.
        """
        self.target_var = target_var
        self.n_samples = n_samples if n_samples is not None else len(perturb_datasets)
        self.use_normal_approx = use_normal_approx

        # Print initial data info
        print(f"\nInitializing with target variable: {target_var}")
        print(f"CTS data dimensions: {cts_data.dims}")
        print(f"Sample perturb dataset dimensions: {perturb_datasets[0].dims}")

        self.process_data(cts_data, perturb_datasets)
        if self.use_normal_approx:
            self._kl_divergence = jax.jit(self._kl_divergence_normal_impl)
        else:
            # Implement other divergence measures if needed
            pass

    def process_data(self, cts_data: xr.Dataset, perturb_datasets: List[xr.Dataset]):
        """
        Process input data by extracting the necessary variables and aligning them.

        Parameters:
        - cts_data: The CTS dataset.
        - perturb_datasets: List of perturbation datasets.
        """
        # Get all unique times from CTS data
        self.times = cts_data.coords['date'].values
        print(f"\nProcessing {len(self.times)} time steps from CTS data")

        # Process CTS data
        self.cts_data = {}
        for time in self.times:
            # Select data for this time
            values = cts_data[self.target_var].sel(date=time).values
            self.cts_data[time] = jnp.array(values)
            print(f"CTS data shape for time {time}: {values.shape}")

        # Process perturbed data
        self.gen_data = {}
        for time in self.times:
            samples = []
            for i, ds in enumerate(perturb_datasets[:self.n_samples]):
                try:
                    # Get data for this time
                    values = ds[self.target_var].sel(date=time).values
                    samples.append(values)
                    if i == 0:
                        print(f"Sample perturb data shape for time {time}: {values.shape}")
                except KeyError:
                    print(f"Time {time} not found in perturb dataset {i}")
                    continue
            if samples:
                self.gen_data[time] = jnp.array(samples)
                print(f"Processed {len(samples)} perturbation samples for time {time}")
            else:
                print(f"No perturbation samples for time {time}")

    @staticmethod
    def _kl_divergence_normal_impl(p_mean, p_std, q_mean, q_std):
        """
        Compute the KL divergence between two normal distributions N(p_mean, p_std^2) and N(q_mean, q_std^2).

        Parameters:
        - p_mean, p_std: Mean and standard deviation of distribution p (generator).
        - q_mean, q_std: Mean and standard deviation of distribution q (CTS).

        Returns:
        - KL divergence value.
        """
        return jnp.log(q_std / p_std) + (p_std ** 2 + (p_mean - q_mean) ** 2) / (2 * q_std ** 2) - 0.5

    def compute_delta_for_timestep(self, time) -> float:
        print(f"\nComputing delta for time: {time}")
        time = np.datetime64(time)
    
        if time not in self.gen_data:
            print(f"Time {time} not found in generator data")
            return np.nan
    
        if time not in self.cts_data:
            print(f"Time {time} not found in CTS data")
            return np.nan
    
        gen_samples = self.gen_data[time]  # Shape: (94,)
        cts_samples = self.cts_data[time]  # Shape: (50,)
    
        print(f"Generator samples shape: {gen_samples.shape}")
        print(f"CTS samples shape: {cts_samples.shape}")
    
        try:
            # Since the data is one-dimensional, we can work directly with the samples
            gen_samples = gen_samples.astype(np.float32)
            cts_samples = cts_samples.astype(np.float32)
    
            # Standardize the samples
            gen_samples_standardized = (gen_samples - gen_samples.mean()) / (gen_samples.std() + 1e-6)
            cts_samples_standardized = (cts_samples - cts_samples.mean()) / (cts_samples.std() + 1e-6)
    
            # Compute means and stds
            gen_mean = jnp.mean(gen_samples_standardized)
            gen_std = jnp.std(gen_samples_standardized) + 1e-6
    
            cts_mean = jnp.mean(cts_samples_standardized)
            cts_std = jnp.std(cts_samples_standardized) + 1e-6
    
            # Compute KL divergence
            kl_value = self._kl_divergence(gen_mean, gen_std, cts_mean, cts_std)
            delta = float(kl_value)
    
            print(f"Computed delta for time {time}: {delta}")
            return delta
    
        except Exception as e:
            print(f"Error computing delta for time {time}: {e}")
            return np.nan


    def compute_delta_timeseries(self) -> pd.DataFrame:
        """
        Compute the error accumulation metric over all time steps.

        Returns:
        - A pandas DataFrame containing the delta values for each time step.
        """
        print("\nComputing delta timeseries")

        deltas = []
        for time in self.times:
            delta = self.compute_delta_for_timestep(time)
            if not np.isnan(delta):
                deltas.append({'time': time, 'delta': delta})
            else:
                print(f"Delta for time {time} is NaN")

        if deltas:
            df = pd.DataFrame(deltas)
            return df
        else:
            print("No valid deltas computed.")
            return pd.DataFrame(columns=['time', 'delta'])

def compute_error_accumulation_jax(
    nwp_data: xr.Dataset,
    perturb_datasets: List[xr.Dataset],
    target_var: str
) -> pd.DataFrame:
    """
    Compute the error accumulation metric using JAX.

    Parameters:
    - nwp_data: xarray.Dataset containing CTS data.
    - perturb_datasets: List of xarray.Datasets containing perturbed data.
    - target_var: The target variable to compute the error on.

    Returns:
    - A pandas DataFrame with the error accumulation metrics.
    """
    print(f"\nStarting computation for target variable: {target_var}")
    print(f"NWP data dimensions: {nwp_data.dims}")
    print(f"Number of perturb datasets: {len(perturb_datasets)}")

    # Configure JAX to use CPU or GPU as needed
    # For CPU: jax.config.update('jax_platform_name', 'cpu')
    # For GPU: jax.config.update('jax_platform_name', 'gpu')

    metric = WeatherErrorAccumulationMetricJAX(
        cts_data=nwp_data,
        perturb_datasets=perturb_datasets,
        target_var=target_var
    )

    return metric.compute_delta_timeseries()
