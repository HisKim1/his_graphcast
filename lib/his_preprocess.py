
import xarray as xr
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import device_put


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


def jax_weighted_mean(ds: xr.Dataset, var_name="2m_temperature"):
    """
    ds     : xarray Dataset, lat/lon 축을 포함
    var_name : ds 내 특정 변수 이름 (예: 'temperature')
    
    Returns
    -------
    가중 평균 결과(단일 값 혹은 필요한 경우 xarray DataArray 형태)
    """
    # (1) Dataset에서 lat, lon 좌표와 원하는 변수 추출
    lat = ds.lat.values  # NumPy array
    lon = ds.lon.values  # NumPy array
    data_np = ds[var_name].values  # NumPy array

    # (2) JAX 디바이스로 보내기 (GPU/TPU 가속 가능)
    lat_j = device_put(jnp.array(lat))
    lon_j = device_put(jnp.array(lon))
    data_j = device_put(jnp.array(data_np))

    # (3) 가중치 계산: w = cos(lat)
    #     lat 축의 shape에 맞춰 브로드캐스팅
    #     만약 (lat, lon) 순으로 데이터가 되어 있다면 아래와 같이 meshgrid로 변환
    #     (lat, lon)의 크기가 각각 (Nlat, Nlon)이라고 가정
    lat2d_j, lon2d_j = jnp.meshgrid(lat_j, lon_j, indexing='ij')
    weights_j = jnp.cos(jnp.deg2rad(lat2d_j))

    # (4) NaN이나 마스킹 처리를 고려해야 한다면, 여기에 조건식 추가
    #     예: data_j = jnp.where(~jnp.isnan(data_j), data_j, 0.0)

    # (5) 가중치 합과 (가중치를 곱한 데이터) 합 구하기
    #     axis=(0,1)은 (lat, lon) 방향으로 합산
    w_sum = jnp.sum(weights_j, axis=(0, 1))
    w_data_sum = jnp.sum(data_j * weights_j, axis=(0, 1))

    # (6) 최종 가중 평균
    weighted_mean_j = w_data_sum / w_sum

    # (7) 결과를 host로 가져오기
    weighted_mean_np = np.array(weighted_mean_j)  # CPU 상의 NumPy 배열

    return weighted_mean_np

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