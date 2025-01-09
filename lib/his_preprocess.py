
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

def jax_weighted_mean_xr(ds: xr.Dataset, var_name=None):
    """
    ds        : xarray Dataset (lat, lon) 축을 반드시 포함
                (추가로 time, level 등의 다른 차원이 있을 수 있음)
    var_name  : ds 내 특정 변수 이름 (예: 'temperature')
                기본값 None인 경우, Dataset 내 모든 변수에 대해 가중평균 계산.

    Returns
    -------
    - var_name가 주어졌다면: xr.DataArray (lat/lon 축이 제거된 형태)
    - var_name가 None이라면: xr.Dataset (모든 변수에 대해 lat/lon 축을 제거한 형태)
    """
    # (0) lat/lon 좌표 추출
    lat_name, lon_name = "lat", "lon"
    lat = ds.coords[lat_name].values  # shape: (lat,)
    lon = ds.coords[lon_name].values  # shape: (lon,)

    # (1) 가중치(numpy) 계산: np.cos(np.deg2rad(lat))
    #     shape = (lat,) -> (1, lat, 1, ...) 로 reshape 필요
    weights_np = np.cos(np.deg2rad(lat))  # (lat,)

    # (2) var_name이 None이면 Dataset 내 모든 변수에 대해 계산
    if var_name is None:
        # 결과를 담을 dict
        data_vars_dict = {}

        for var in ds.data_vars:
            da = ds[var]

            # lat/lon이 꼭 데이터 차원에 있어야 가중평균 가능
            if (lat_name not in da.dims) or (lon_name not in da.dims):
                # lat, lon 축이 없다면 그냥 원본을 넣거나
                # 혹은 에러 처리(필요에 따라)
                data_vars_dict[var] = da
                continue

            # --- 아래 로직은 단일 변수에 대해 가중평균 계산하는 부분 ---
            data_np = da.values
            original_dims = da.dims
            other_dims = [d for d in original_dims if d not in [lat_name, lon_name]]

            # reshape for broadcast
            reshape_dims = [1] * da.ndim
            lat_idx = original_dims.index(lat_name)
            lon_idx = original_dims.index(lon_name)
            reshape_dims[lat_idx] = len(lat)
            # lon_idx 위치에는 1만 유지( weights = cos(lat) )
            w_reshaped = weights_np.reshape(reshape_dims)

            # JAX로 변환
            data_j = device_put(jnp.array(data_np))
            weights_j = device_put(jnp.array(w_reshaped))

            # lat/lon 축 인덱스 구하기
            axes_to_sum = (lat_idx, lon_idx)

            # 가중합
            w_sum = jnp.sum(weights_j, axis=axes_to_sum)
            w_data_sum = jnp.sum(data_j * weights_j, axis=axes_to_sum)
            weighted_mean_j = w_data_sum / w_sum

            weighted_mean_np = np.array(weighted_mean_j)

            # xarray DataArray로 복원
            coords_dict = {d: ds.coords[d] for d in other_dims}
            da_out = xr.DataArray(
                data=weighted_mean_np,
                coords=coords_dict,
                dims=other_dims,
                name=f"{var}_weighted_mean"
            )

            data_vars_dict[var] = da_out

        # 모든 변수에 대한 결과를 Dataset으로 결합
        ds_out = xr.Dataset(data_vars=data_vars_dict)
        return ds_out

    else:
        # 단일 변수 계산
        da = ds[var_name]
        original_dims = da.dims
        other_dims = [d for d in original_dims if d not in [lat_name, lon_name]]

        data_np = da.values

        # reshape for broadcast
        reshape_dims = [1] * da.ndim
        lat_idx = original_dims.index(lat_name)
        lon_idx = original_dims.index(lon_name)
        reshape_dims[lat_idx] = len(lat)
        w_reshaped = weights_np.reshape(reshape_dims)

        data_j = device_put(jnp.array(data_np))
        weights_j = device_put(jnp.array(w_reshaped))

        axes_to_sum = (lat_idx, lon_idx)
        w_sum = jnp.sum(weights_j, axis=axes_to_sum)
        w_data_sum = jnp.sum(data_j * weights_j, axis=axes_to_sum)
        weighted_mean_j = w_data_sum / w_sum
        weighted_mean_np = np.array(weighted_mean_j)

        # xarray DataArray로 복원
        coords_dict = {d: ds.coords[d] for d in other_dims}
        da_out = xr.DataArray(
            data=weighted_mean_np,
            coords=coords_dict,
            dims=other_dims,
            name=f"{var_name}_weighted_mean"
        )
        return da_out


def jax_weighted_mean(ds: xr.Dataset, var_name="2m_temperature"):
    """
    ds     : xarray Dataset, lat/lon 축을 포함
    var_name : ds 내 특정 변수 이름 (예: 'temperature')
    
    Returns
    -------
    가중 평균 결과(단일 값 혹은 필요한 경우 xarray DataArray 형태)
    """
    lat = ds.lat.values  # NumPy array
    lon = ds.lon.values  # NumPy array
    data_np = ds[var_name].values  # NumPy array

    lat_j = device_put(jnp.array(lat))
    lon_j = device_put(jnp.array(lon))
    data_j = device_put(jnp.array(data_np))
    
    lat2d_j, lon2d_j = jnp.meshgrid(lat_j, lon_j, indexing='ij')
    weights_j = jnp.cos(jnp.deg2rad(lat2d_j))

    # (4) NaN이나 마스킹 처리를 고려해야 한다면, 여기에 조건식 추가
    #     예: data_j = jnp.where(~jnp.isnan(data_j), data_j, 0.0)

    # (5) 가중치 합과 (가중치를 곱한 데이터) 합 구하기
    #     axis=(0,1)은 (lat, lon) 방향으로 합산
    w_sum = jnp.sum(weights_j, axis=(0, 1))
    w_data_sum = jnp.sum(data_j * weights_j, axis=(0, 1))

    weighted_mean_j = w_data_sum / w_sum

    weighted_mean_np = np.array(weighted_mean_j)

    return weighted_mean_np

def weighted_mean(data, var_name=None):
    if isinstance(data, xr.Dataset) or isinstance(data, xr.DataArray):
        weights = np.cos(np.deg2rad(data.lat))
        weights.name = "weights"
        if var_name:
            weighted = data[var_name].weighted(weights)
            return weighted.mean(('lat', 'lon'))
        else:
            weighted = data.weighted(weights)
            return weighted.mean(('lat', 'lon'))
    else:
        raise TypeError("Input must be an xarray.Dataset or xarray.DataArray")

def preprocess_GC(dataset:xr.Dataset, target_var="2m_temperature", region=None):
    if target_var == "2m_temperature":
        dataset = dataset.resample(time="1D").mean().squeeze('batch')
        dataset["time"] = pd.date_range("2021-06-22", periods=10, freq="1D")
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