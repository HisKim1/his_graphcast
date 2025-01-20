
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



def jax_weighted_mean(ds: xr.Dataset, var_name=None):
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
    lat_name, lon_name = "lat", "lon"
    lat = ds.coords[lat_name].values  # shape: (lat,)
    lon = ds.coords[lon_name].values  # shape: (lon,)

    # (1) 가중치(numpy) 계산: np.cos(np.deg2rad(lat))
    weights_np = np.cos(np.deg2rad(lat))  # shape (lat,)

    # ------------------------------------------------
    # (추가) xarray의 weighted.mean(skipna=True)와 동일하게 NaN 처리를 위해
    #        실제 데이터 계산 시 NaN인 곳은 가중치도 제외(0으로)하도록 마스킹
    # ------------------------------------------------

    def _compute_weighted_mean(da: xr.DataArray):
        original_dims = da.dims
        other_dims = [d for d in original_dims if d not in [lat_name, lon_name]]

        data_np = da.values  # (time, batch, lat, lon, ...) 등등

        # reshape weights for broadcast
        reshape_dims = [1] * da.ndim
        lat_idx = original_dims.index(lat_name)
        lon_idx = original_dims.index(lon_name)
        reshape_dims[lat_idx] = len(lat)

        # lon_idx 위치에는 1을 유지 -> (lat, 1) 형태가 되어
        # data_np와 곱셈할 때 (lat, lon)을 broadcast하게 됨
        w_reshaped = weights_np.reshape(reshape_dims)

        data_j = device_put(jnp.array(data_np))
        weights_j = device_put(jnp.array(w_reshaped))

        # --- NaN 마스킹 ---
        # data가 NaN이면, 그 격자의 weights도 0으로 만들기
        valid_mask = ~jnp.isnan(data_j)
        data_j = jnp.where(valid_mask, data_j, 0.0)
        weights_j = jnp.where(valid_mask, weights_j, 0.0)

        # lat/lon 합산
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
            name=f"{da.name}_weighted_mean"
        )
        return da_out

    # (2) var_name이 None이면 Dataset 내 모든 변수에 대해 계산
    if var_name is None:
        data_vars_dict = {}
        for var in ds.data_vars:
            da = ds[var]
            # lat, lon 축이 있으면 가중평균 계산
            if (lat_name in da.dims) and (lon_name in da.dims):
                da_out = _compute_weighted_mean(da)
                data_vars_dict[var] = da_out
            else:
                # lat, lon 축이 없는 변수는 원본 그대로 저장 (혹은 제외)
                data_vars_dict[var] = da
        ds_out = xr.Dataset(data_vars=data_vars_dict)
        return ds_out

    else:
        da = ds[var_name]
        return _compute_weighted_mean(da)

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