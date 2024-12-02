
import xarray as xr
import numpy as np
import pandas as pd


PRESSURE_VARIABLES=[
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "temperature",
    "vertical_velocity"
]

SURFACE_VARIABLES=[
    "10m_u_component_of_wind", 
    "10m_v_component_of_wind", 
    "total_precipitation_6hr", 
    "mean_sea_level_pressure",
]

def weighted_mean(dataset: xr.Dataset):
    weights = np.cos(np.deg2rad(dataset.lat))
    weights.name = "weights"
    weighted = dataset.weighted(weights)
    
    return weighted.mean(('lat', 'lon'))


def preprocess_GC(dataset:xr.Dataset, target_var="2m_temperature", region=None):
    if target_var == "2m_temperature":
        dataset = dataset.resample(time="1D").mean().squeeze('batch')
        dataset["time"] = pd.date_range("2021-06-22", periods=7, freq="1D")
        dataset = dataset.rename({"time":"date"})
        dataset = dataset.drop_vars(PRESSURE_VARIABLES + SURFACE_VARIABLES + ['level', 'geopotential'])

    elif target_var == "geopotential":
        dataset = dataset.sel(level=500).resample(time="1D").mean().squeeze('batch')
        dataset["time"] = pd.date_range("2021-06-22", periods=7, freq="1D")
        dataset = dataset.rename({"time":"date"})
        dataset = dataset.drop_vars(PRESSURE_VARIABLES + SURFACE_VARIABLES + ['level', '2m_temperature'])

    # TODO: add region selection 
    if region:
        dataset = dataset.sel(lat=slice(25, 60), lon=slice(102.5, 150))

    return dataset


def preprocess_nwp(dataset:xr.Dataset, target_var="2m_temperature", region=None):
    dataset = dataset.expand_dims(dim={'date': [dataset.time.values[0]]}).compute()
    dataset = dataset.rename({'time': 'ensemble'})
    dataset['ensemble'] = np.arange(1, 51)

    if target_var == "2m_temperature":
        dataset = dataset.drop_vars('height')

    elif target_var == "geopotential":
        dataset.gh.attrs['units'] = 'm^2/s^2'
        dataset.gh.attrs['long_name'] = 'Geopotential'
        dataset = dataset.assign_coords(lev=dataset.lev / 100)
        dataset.lev.attrs['units'] = 'hPa'
        dataset['gh'] = dataset['gh'] * 9.80665
        dataset = dataset.rename({'gh':'geopotential', 'lev':'level'})
        dataset = dataset.sel(level=500)

    # TODO: add region selection 
    if region:
        dataset = dataset.sel(lat=slice(60, 24), lon=slice(102, 150))

    return dataset

def preprocess_era(dataset:xr.Dataset, target_var="2m_temperature", region=None):
    if region:
        dataset = dataset.sel(lat=slice(60, 25), lon=slice(102.5, 150))