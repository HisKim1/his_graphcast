import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphcast import data_utils

# TODO: original은 들어오기 전에 squeeze batch해줘야 함!
def add_perturbation(original: xr.Dataset, 
                     variables: list[str], 
                     scale: float, 
                     perturb_timestep: list = [0, 1]):
    if "batch" in original.dims:
        original = original.squeeze("batch")

    with xr.open_dataset(f'testdata/stats/40yr_std_daily_4var.nc') as std: 
        perturbed = original.copy()
        
        for i in perturb_timestep:
            idx_time = original.time.isel(time=0).values.astype(np.int64)//21600
            normal_dist = np.random.normal(loc=0, scale=1, size=original[variables[0]].isel(time=i).shape)
            normal_dist = xr.DataArray(
                data=normal_dist,
                dims=('lat','lon'),
                coords={
                    'lat': original.lat,
                    'lon': original.lon
                }
            )
    
            for var in variables:
                perturbed.isel(time=i)[var] = original[var].isel(time=i) + scale * (normal_dist * std[var].isel(hour=idx_time))
                
        if 'perturbation' in list(perturbed.data_vars):
            perturbed = perturbed.drop_vars('perturbation')

    perturbed = perturbed.transpose('time', 'lat', 'lon', 'level').expand_dims('batch')
    if 'batch' not in perturbed['datetime'].dims:
        perturbed['datetime'] = perturbed['datetime'].expand_dims({'batch': [0]}, axis=0)

    for var in ['geopotential_at_surface', 'land_sea_mask']:
        if var in perturbed and 'batch' in perturbed[var].dims:
            perturbed[var] = perturbed[var].squeeze('batch')
    return perturbed

def convert_scale(dataset):
    """
        convert scale of tp_6hr and slp to mm and hPa
    """
    if 'total_precipitation_6hr' in dataset:
        dataset['total_precipitation_6hr'] *= 1000
        dataset['total_precipitation_6hr'].attrs['units'] = 'mm'

    if 'mean_sea_level_pressure' in dataset:
        dataset['mean_sea_level_pressure'] /= 100
        dataset['mean_sea_level_pressure'].attrs['units'] = 'hPa'

    return dataset


def transform_dataset(dataset):
    """
        Transform the dataset to the format such as variable names and coordinates required by the model.
        Referenced at data_concat.ipynb
    """
    # 1. batch 차원 추가
    if 'batch' not in dataset.dims:
        dataset = dataset.expand_dims(dim={'batch': [0]})
    
    # 2. 변수 이름 매핑
    name_mapping = {
        'u10': '10m_u_component_of_wind',
        'v10': '10m_v_component_of_wind',
        't2m': '2m_temperature',
        'msl': 'mean_sea_level_pressure',
        'tp': 'total_precipitation_6hr',
        'z': 'geopotential',
        'q': 'specific_humidity',
        't': 'temperature',
        'u': 'u_component_of_wind',
        'v': 'v_component_of_wind',
        'w': 'vertical_velocity',
        'tisr': 'toa_incident_solar_radiation',
        'lsm': 'land_sea_mask',
        'longitude': 'lon',
        'latitude': 'lat'
    }
    
    dataset = dataset.rename({k: v for k, v in name_mapping.items() 
                              if k in dataset.variables or k in dataset.coords})

    # 4. time 좌표를 timedelta로 변환
    start_time = dataset.time.values[0]
    dataset['time'] = (dataset.time - start_time).astype('timedelta64[ns]')
    
    # 5. datetime 좌표 추가
    dataset.coords['datetime'] = ('time', pd.date_range(start=start_time, periods=len(dataset.time), freq='6h'))
    dataset['datetime'] = dataset['datetime'].expand_dims({'batch': [0]}, axis=0)

    for var in ['geopotential_at_surface', 'land_sea_mask']:
        if var in dataset:
            dataset[var] = dataset[var].isel(time=0, drop=True)
            if 'batch' in dataset[var].dims:
                dataset[var] = dataset[var].squeeze('batch')
    
    return dataset.reindex(lat=dataset.lat[::-1])

def create_forcing_dataset(time_steps, resolution, start_time):
    lon = np.arange(0.0, 360.0, resolution, dtype=np.float32)
    lat = np.arange(-90.0, 90.0 + resolution/2, resolution, dtype=np.float32)
    
    start_datetime = pd.to_datetime(start_time) + pd.Timedelta(hours=12)
    datetime = pd.date_range(start=start_datetime, periods=time_steps, freq='6h')
    
    time = pd.timedelta_range(start='6h', periods=time_steps, freq='6h')
    
    # Create the dataset
    ds = xr.Dataset(
        coords={
            'lon': ('lon', lon),
            'lat': ('lat', lat),
            'datetime': ('datetime', datetime),
            'time': ('time', time)
        }
    )
    
    ds.lat.attrs['long_name'] = 'latitude'
    ds.lat.attrs['units'] = 'degrees_north'
    
    ds.lon.attrs['long_name'] = 'longitude'
    ds.lon.attrs['units'] = 'degrees_east'
    
    variables = ['toa_incident_solar_radiation',
                 'year_progress_sin',
                 'year_progress_cos',
                 'day_progress_sin',
                 'day_progress_cos']
    
    data_utils.add_derived_vars(ds)
    data_utils.add_tisr_var(ds)
    
     # `datetime` is needed by add_derived_vars but breaks autoregressive rollouts.
    ds = ds.drop_vars("datetime")
    
    ds = ds[list(variables)]
    
    # 각 변수에 'batch' 차원 추가
    for var in variables:
        # 현재 변수의 차원과 데이터 가져오기
        current_dims = ds[var].dims
        current_data = ds[var].values

        # 'batch' 차원을 추가한 새로운 데이터 배열 생성
        new_shape = (1,) + current_data.shape
        perturbed = np.zeros(new_shape, dtype=current_data.dtype)
        perturbed[0] = current_data

        # 새로운 차원 순서 정의 ('batch'를 첫 번째로)
        new_dims = ('batch',) + current_dims

        # 새로운 DataArray 생성 및 할당 (coordinate는 추가하지 않음)
        ds[var] = xr.DataArray(
            data=perturbed,
            dims=new_dims,
            coords={dim: ds[dim] for dim in current_dims}  # 'batch'는 coordinate에 포함하지 않음
        )

    return ds

def create_target_dataset(time_steps, resolution, pressure_levels):
    # Define coordinates
    lon = np.arange(0.0, 360.0, resolution, dtype=np.float32)
    lat = np.arange(-90.0, 90.0 + resolution/2, resolution, dtype=np.float32)
    
    if pressure_levels == 37:
        level = [   1,    2,    3,    5,    7,   10,   20,   30,   50,   70,  100,  125, 150,  175,  200,  225,  250,  300,  350,  400,  450,  500,  550,  600, 650,  700,  750,  775,  800,  825,  850,  875,  900,  925,  950,  975, 1000]
    elif pressure_levels == 13:
        level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    else:
        raise ValueError("Unsupported number of pressure levels. Choose either 37 or 13.")

    level = np.array(level, dtype=np.int64)

    # 시작 시간부터 time_steps 개의 6시간 간격 타임델타 생성
    time = pd.timedelta_range(start='6h', periods=time_steps, freq='6h')
    # timedelta64[ns]로 명시적 변환
    # time = time.astype('timedelta64[ns]')/

    # Create the dataset
    ds = xr.Dataset(
        coords={
            'lon': ('lon', lon),
            'lat': ('lat', lat),
            'level': ('level', level.astype(np.int32)),
            'time': ('time', time),
        }
    )

    ds.lat.attrs['long_name'] = 'latitude'
    ds.lat.attrs['units'] = 'degrees_north'

    ds.lon.attrs['long_name'] = 'longitude'
    ds.lon.attrs['units'] = 'degrees_east'

    # Add data variables filled with NaN
    surface_vars = ['2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind', 
                    '10m_u_component_of_wind', 'total_precipitation_6hr']
    level_vars = ['temperature', 'geopotential', 'u_component_of_wind', 
                  'v_component_of_wind', 'vertical_velocity', 'specific_humidity']

    for var in surface_vars:
        ds[var] = xr.DataArray(
            data=np.full((1, time_steps, len(lat), len(lon)), np.nan, dtype=np.float32),
            dims=['batch', 'time', 'lat', 'lon']
        )

    for var in level_vars:
        ds[var] = xr.DataArray(
            data=np.full((1, time_steps, len(level), len(lat), len(lon)), np.nan, dtype=np.float32),
            dims=['batch', 'time', 'level', 'lat', 'lon'],
        )

    ds = ds.transpose("batch", "time", "lat", "lon", ...)

    return ds

def compare_dataarrays(da1: xr.DataArray, da2: xr.DataArray):
    differences = []

    # Compare names
    if da1.name != da2.name:
        differences.append(f"Names differ: {da1.name} != {da2.name}")

    # Compare dimensions
    if da1.dims != da2.dims:
        differences.append(f"Dimensions differ: {da1.dims} != {da2.dims}")
    
    # Compare coordinates
    da1_coords = set(da1.coords.keys())
    da2_coords = set(da2.coords.keys())
    if da1_coords != da2_coords:
        differences.append(f"Coordinate keys differ: {da1_coords} != {da2_coords}")
    else:
        for coord in da1.coords:
            if not da1.coords[coord].identical(da2.coords[coord]):
                differences.append(f"Coordinate '{coord}' values differ:")
                differences.append(f"{da1.coords[coord]-da2.coords[coord]}")

    # Compare attributes
    if da1.attrs != da2.attrs:
        differences.append(f"Attributes differ:")
        differences.append(f"  DA1 attrs: {da1.attrs}")
        differences.append(f"  DA2 attrs: {da2.attrs}")

    # Compare shapes
    if da1.shape != da2.shape:
        differences.append(f"Shapes differ: DA1 {da1.shape} != DA2 {da2.shape}")
    
    # Compare dtypes
    if da1.dtype != da2.dtype:
        differences.append(f"Dtypes differ: DA1 {da1.dtype} != DA2 {da2.dtype}")
    
    # Compare values
    if np.prod(da1.shape) <= 100:  # If array is small, compare all values
        # Use pandas for element-wise comparison with NaN handling
        df1 = pd.DataFrame(da1.values.ravel())
        df2 = pd.DataFrame(da2.values.ravel())
        if not df1.equals(df2):
            differences.append(f"Values differ:")
            differences.append(f"  DA1: {da1.values}")
            differences.append(f"  DA2: {da2.values}")
            # Show where the differences are (including NaN disagreements)
            diff_mask = ~(df1.eq(df2) | (df1.isna() & df2.isna()))
            diff_indices = np.where(diff_mask)[0]
            differences.append(f"  Differing indices: {diff_indices}")
            differences.append(f"  DA1 values at those indices: {df1.iloc[diff_indices].values.ravel()}")
            differences.append(f"  DA2 values at those indices: {df2.iloc[diff_indices].values.ravel()}")
    else:  # For large arrays, compare statistics and a sample
        da1_stats = da1.mean().item(), da1.std().item(), np.sum(np.isnan(da1.values))
        da2_stats = da2.mean().item(), da2.std().item(), np.sum(np.isnan(da2.values))
        if not np.allclose(da1_stats[:2], da2_stats[:2], equal_nan=True) or da1_stats[2] != da2_stats[2]:
            differences.append(f"Statistics differ:")
            differences.append(f"  DA1 mean, std, NaN count: {da1_stats}")
            differences.append(f"  DA2 mean, std, NaN count: {da2_stats}")
        
        # Compare a sample of values
        sample_indices = tuple(np.random.randint(0, s, 5) for s in da1.shape)
        da1_sample = da1.isel(dict(zip(da1.dims, sample_indices))).values
        da2_sample = da2.isel(dict(zip(da2.dims, sample_indices))).values
        if not np.array_equal(da1_sample, da2_sample, equal_nan=True):
            differences.append(f"Sample values differ at indices {sample_indices}:")
            differences.append(f"  DA1: {da1_sample}")
            differences.append(f"  DA2: {da2_sample}")

    if not differences:
        print("DataArrays are identical")
    else:
        for difference in differences:
            print(difference)

# input & forcing에서 잘 먹히는지는 확인 안 됨
def compare_datasets4target(ds1: xr.Dataset, ds2: xr.Dataset):
    differences = []

    # Compare dimensions
    if ds1.dims != ds2.dims:
        differences.append(f"Dimensions differ: {ds1.dims} != {ds2.dims}")
    
    # Compare coordinates
    ds1_coords = set(ds1.coords.keys())
    ds2_coords = set(ds2.coords.keys())
    if ds1_coords != ds2_coords:
        differences.append(f"Coordinate keys differ: {ds1_coords} != {ds2_coords}")
    else:
        for coord in ds1.coords:
            if not ds1.coords[coord].identical(ds2.coords[coord]):
                differences.append(f"Coordinate '{coord}' values differ:")
                differences.append(f"  DS1: {ds1.coords[coord].values}")
                differences.append(f"  DS2: {ds2.coords[coord].values}")

    # Compare data variables
    ds1_vars = set(ds1.data_vars.keys())
    ds2_vars = set(ds2.data_vars.keys())
    if ds1_vars != ds2_vars:
        differences.append(f"Data variable keys differ:")
        differences.append(f"  Only in DS1: {ds1_vars - ds2_vars}")
        differences.append(f"  Only in DS2: {ds2_vars - ds1_vars}")
    
    common_vars = ds1_vars.intersection(ds2_vars)
    for var in common_vars:
        if not ds1.data_vars[var].identical(ds2.data_vars[var]):
            differences.append(f"Data variable '{var}' differs:")
            
            # Compare shapes
            if ds1[var].shape != ds2[var].shape:
                differences.append(f"  Shape differs: DS1 {ds1[var].shape} != DS2 {ds2[var].shape}")
            
            # Compare dtypes
            if ds1[var].dtype != ds2[var].dtype:
                differences.append(f"  Dtype differs: DS1 {ds1[var].dtype} != DS2 {ds2[var].dtype}")
            
            # Compare values (for small arrays or samples for large arrays)
            if np.prod(ds1[var].shape) <= 100:  # If array is small, compare all values
                # Use pandas for element-wise comparison with NaN handling
                df1 = pd.DataFrame(ds1[var].values.ravel())
                df2 = pd.DataFrame(ds2[var].values.ravel())
                if not df1.equals(df2):
                    differences.append(f"  Values differ:")
                    differences.append(f"    DS1: {ds1[var].values}")
                    differences.append(f"    DS2: {ds2[var].values}")
                    # Show where the differences are (including NaN disagreements)
                    diff_mask = ~(df1.eq(df2) | (df1.isna() & df2.isna()))
                    diff_indices = np.where(diff_mask)[0]
                    differences.append(f"    Differing indices: {diff_indices}")
                    differences.append(f"    DS1 values at those indices: {df1.iloc[diff_indices].values.ravel()}")
                    differences.append(f"    DS2 values at those indices: {df2.iloc[diff_indices].values.ravel()}")
            else:  # For large arrays, compare statistics and a sample
                ds1_stats = ds1[var].mean().item(), ds1[var].std().item(), np.sum(np.isnan(ds1[var].values))
                ds2_stats = ds2[var].mean().item(), ds2[var].std().item(), np.sum(np.isnan(ds2[var].values))
                if not np.allclose(ds1_stats[:2], ds2_stats[:2], equal_nan=True) or ds1_stats[2] != ds2_stats[2]:
                    differences.append(f"  Statistics differ:")
                    differences.append(f"    DS1 mean, std, NaN count: {ds1_stats}")
                    differences.append(f"    DS2 mean, std, NaN count: {ds2_stats}")
                
                # Compare a sample of values
                sample_indices = tuple(np.random.randint(0, s, 5) for s in ds1[var].shape)
                ds1_sample = ds1[var].isel(dict(zip(ds1[var].dims, sample_indices))).values
                ds2_sample = ds2[var].isel(dict(zip(ds2[var].dims, sample_indices))).values
                if not np.array_equal(ds1_sample, ds2_sample, equal_nan=True):
                    differences.append(f"  Sample values differ at indices {sample_indices}:")
                    differences.append(f"    DS1: {ds1_sample}")
                    differences.append(f"    DS2: {ds2_sample}")

    if not differences:
        print("Datasets are identical")
    else:
        for difference in differences:
            print(difference)
