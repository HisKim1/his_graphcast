import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphcast import data_utils

import numpy as np
import xarray as xr

import jax.numpy as jnp
from jax import random, device_put
import jax

jax.config.update("jax_platform_name", "gpu")

REGION_BOUNDARIES = {
    "global": {"lat": (-90, 90), "lon": (0, 360)},
    "Northern Hemisphere Extra-tropics": {"lat": (20, 90)},
    "Southern Hemisphere Extra-tropics": {"lat": (-90, -20)},
    "Tropics": {"lat": (-20, 20)},
    "Arctic": {"lat": (60, 90)},
    "Antarctic": {"lat": (-90, -60)},
    "Europe": {"lat": (35, 75), "lon": (-12.5, 42.5)},
    "North America": {"lat": (25, 60), "lon": (-120, -75)},
    "North Atlantic": {"lat": (25, 60), "lon": (-70, -20)},
    "North Pacific": {"lat": (25, 60), "lon": (145, -130)},
    "East Asia": {"lat": (25, 60), "lon": (102.5, 150)},
    "AusNZ": {"lat": (-45, -12.5), "lon": (120, 175)}
}

SURFACE_VARIABLE = ['10m_u_component_of_wind',
 '10m_v_component_of_wind',
 '2m_temperature',
 'mean_sea_level_pressure',
 'total_precipitation_6hr'] # cannot be negative

PRESSURE_VARIABLE = ['geopotential',
 'specific_humidity', # cannot be negative
 'temperature',
 'u_component_of_wind',
 'v_component_of_wind',
 'vertical_velocity']

def add_proportional_perturbation(
    original: xr.Dataset,
    variables: list[str],
    scale: float = 1,
    perturb_timestep: list[int] = [0, 1],
    num_points: int = 23568048,
    wipe_out: bool = True,
    region: dict = None,
):
    """
    original : xr.Dataset
        - (batch, time, lat, lon, [level]) 구조의 원본 데이터셋
    variables : list[str]
        - perturb를 적용할 변수 목록
        - 예) SURFACE_VARIABLE + PRESSURE_VARIABLE
    scale : float
        - (기존) gaussian noise의 scale factor (wipe_out=False 일 때 사용)
        - (수정) 자기 자신의 값 * scale * Normal(0,1)을 추가
    perturb_timestep : list[int]
        - perturb를 적용할 time index 리스트 (기본값: [0,1])
    num_points : int
        - 전체 (표면+대기압) 격자의 10%에 해당하는 개수 (기본값: 23568048)
    wipe_out : bool
        - True이면 mean 값으로 치환, False이면 자기 자신 * scale * N(0,1)을 더함
    region : dict
        - 특정 영역에서만 perturb를 주고 싶을 때 사용.
        - 예: region = {'lat': (lat_min, lat_max), 'lon': (lon_min, lon_max)}
        - None이면 전 지구 영역을 대상으로 perturb를 적용.
    """
    NONNEGATIVE_VARS = {"total_precipitation_6hr", "specific_humidity"}

    # ---- 사전 세팅
    perturbed = original.copy()

    # lat/lon 좌표값
    lat_vals = perturbed["lat"].values
    lon_vals = perturbed["lon"].values

    batch_size = perturbed.sizes.get('batch', 1)
    time_size = perturbed.sizes.get('time', 1)

    # region이 주어졌다면, 그 영역 내의 lat/lon 인덱스만 추출
    if region is not None:
        # region['lat'] = (lat_min, lat_max), region['lon'] = (lon_min, lon_max)
        lat_min, lat_max = region['lat']
        lon_min, lon_max = region['lon']

        # 해당 영역에 속하는 lat/lon 인덱스 마스킹
        lat_mask = (lat_vals >= lat_min) & (lat_vals <= lat_max)
        lon_mask = (lon_vals >= lon_min) & (lon_vals <= lon_max)

        # region 내 lat, lon 인덱스
        lat_idx_region = np.where(lat_mask)[0]  # 예: [10, 11, 12, ...] 형태
        lon_idx_region = np.where(lon_mask)[0]

        region_lat_size = len(lat_idx_region)
        region_lon_size = len(lon_idx_region)
    else:
        # region이 None이면 전 영역 사용
        lat_idx_region = np.arange(perturbed.sizes['lat'])
        lon_idx_region = np.arange(perturbed.sizes['lon'])
        region_lat_size = perturbed.sizes['lat']
        region_lon_size = perturbed.sizes['lon']

    # level 크기 (level 없는 경우 1로 처리)
    level_size = perturbed.sizes['level'] if 'level' in perturbed.sizes else 1

    # surface/pressure 변수 구분
    surface_vars = []
    pressure_vars = []
    for var in variables:
        if 'level' in perturbed[var].dims:
            pressure_vars.append(var)
        else:
            surface_vars.append(var)

    # (1) region 내부에서 변수별 격자 개수 계산
    #     surface: region_lat_size * region_lon_size
    #     pressure: level_size * region_lat_size * region_lon_size
    surf_grid = region_lat_size * region_lon_size
    press_grid = level_size * region_lat_size * region_lon_size

    var_sizes = []
    for var in surface_vars:
        var_sizes.append(surf_grid)
    for var in pressure_vars:
        var_sizes.append(press_grid)

    # jnp.array로 변환
    var_sizes_j = jnp.array(var_sizes)  # shape=(len(variables),)

    # var_offsets: 누적합 (0, var_sizes[0], var_sizes[0]+var_sizes[1], ...)
    var_offsets_j = jnp.cumsum(var_sizes_j)  # shape=(len(variables),)
    var_offsets_j = jnp.concatenate([jnp.array([0]), var_offsets_j])  # shape=(len(variables)+1,)

    # (2) JAX로 데이터를 옮겨놓기
    original_data = {}
    for var in variables:
        original_data[var] = device_put(jnp.array(perturbed[var].values))

    # 표준편차/평균 데이터셋 (wipe_out을 위해 평균은 계속 사용)
    # std는 더 이상 사용하지 않으므로 불러오지 않아도 되지만, 혹시 필요하면 주석만 처리
    with xr.open_dataset("/geodata2/S2S/DL/GC_input/stat/stats_mean_by_level.nc") as mean_ds:
        mean_data = {}
        for var in variables:
            mean_data[var] = device_put(jnp.array(mean_ds[var].values))

        # 랜덤 시드
        key = random.PRNGKey(42)

        # (3) time loop
        for t_idx in perturb_timestep:
            if t_idx >= time_size:
                continue  # time 범위 넘어가면 skip

            # batch loop
            for b_idx in range(batch_size):
                # 3-1) num_points개를 region 내 ( var_sizes 합 )에서 무작위로 뽑기
                total_points_region = var_offsets_j[-1]  # region 내 모든 변수의 격자 합
                key, subkey = random.split(key)
                rand_indices = random.choice(
                    subkey,
                    total_points_region,
                    shape=(num_points,),
                    replace=True  # region 범위 내에서 중복 허용
                )

                # 3-2) 각 랜덤 인덱스가 어떤 변수인지 searchsorted로 판별
                var_id = jnp.searchsorted(var_offsets_j, rand_indices, side='right') - 1
                local_index = rand_indices - var_offsets_j[var_id]

                # 3-3) 변수별로 모아서 perturb
                for i, var in enumerate(surface_vars + pressure_vars):
                    mask = (var_id == i)
                    count_i = jnp.sum(mask)
                    if count_i == 0:
                        continue

                    key, subkey = random.split(key)
                    normal_dist = random.normal(subkey, shape=(count_i,))

                    if var in surface_vars:
                        # unravel
                        local_index_surf = local_index[mask]
                        lat_idx_local, lon_idx_local = jnp.divmod(local_index_surf, region_lon_size)

                        # 실제 global lat/lon 인덱스로 변환
                        lat_idx_global = jnp.array(lat_idx_region)[lat_idx_local]
                        lon_idx_global = jnp.array(lon_idx_region)[lon_idx_local]

                        if wipe_out:
                            # (이전 로직과 동일) 평균값으로 치환
                            original_data[var] = original_data[var].at[
                                b_idx, t_idx, lat_idx_global, lon_idx_global
                            ].set(mean_data[var])
                        else:
                            # **자기 자신의 값 * scale * normal_dist**를 더함
                            old_vals = original_data[var][b_idx, t_idx, lat_idx_global, lon_idx_global]
                            noise = old_vals * scale * normal_dist
                            original_data[var] = original_data[var].at[
                                b_idx, t_idx, lat_idx_global, lon_idx_global
                            ].add(noise)

                        # 음수 불가 변수는 clip 처리
                        if var in NONNEGATIVE_VARS:
                            updated_vals = original_data[var][b_idx, t_idx, lat_idx_global, lon_idx_global]
                            clipped_vals = jnp.clip(updated_vals, 0.0, None)  # 최소 0
                            original_data[var] = original_data[var].at[
                                b_idx, t_idx, lat_idx_global, lon_idx_global
                            ].set(clipped_vals)

                    else:
                        # pressure var
                        local_index_press = local_index[mask]
                        lvl_idx_local = local_index_press // (region_lat_size * region_lon_size)
                        latlon_local = local_index_press % (region_lat_size * region_lon_size)
                        lat_idx_local, lon_idx_local = jnp.divmod(latlon_local, region_lon_size)

                        # 실제 global lat/lon 인덱스로 변환
                        lat_idx_global = jnp.array(lat_idx_region)[lat_idx_local]
                        lon_idx_global = jnp.array(lon_idx_region)[lon_idx_local]

                        if wipe_out:
                            # (이전 로직과 동일) 평균값으로 치환
                            original_data[var] = original_data[var].at[
                                b_idx, t_idx, lvl_idx_local, lat_idx_global, lon_idx_global
                            ].set(mean_data[var][lvl_idx_local])
                        else:
                            # **자기 자신의 값 * scale * normal_dist**를 더함
                            old_vals = original_data[var][b_idx, t_idx, lvl_idx_local, lat_idx_global, lon_idx_global]
                            noise = old_vals * scale * normal_dist
                            original_data[var] = original_data[var].at[
                                b_idx, t_idx, lvl_idx_local, lat_idx_global, lon_idx_global
                            ].add(noise)

                        # 음수 불가 변수는 clip 처리
                        if var in NONNEGATIVE_VARS:
                            updated_vals = original_data[var][b_idx, t_idx, lvl_idx_local, lat_idx_global, lon_idx_global]
                            clipped_vals = jnp.clip(updated_vals, 0.0, None)
                            original_data[var] = original_data[var].at[
                                b_idx, t_idx, lvl_idx_local, lat_idx_global, lon_idx_global
                            ].set(clipped_vals)

        # (4) perturb된 결과를 numpy array로 되돌려서 xarray에 붙여넣기
        for var in variables:
            perturbed[var].values = np.array(original_data[var])

        # 혹시 'perturbation' 변수가 있었다면 제거
        if 'perturbation' in perturbed.data_vars:
            perturbed = perturbed.drop_vars('perturbation')

    return perturbed


def add_regional_shuffle_perturbation(
    original: xr.Dataset,
    variables: list[str],
    scale: float = 1,
    perturb_timestep: list[int] = [0, 1],
    num_points: int = 23568048,
    wipe_out: bool = True,
    region: dict = None,
):
    """
    original : xr.Dataset
        - (batch, time, lat, lon, [level]) 구조의 원본 데이터셋
    variables : list[str]
        - perturb를 적용할 변수 목록
        - 예) SURFACE_VARIABLE + PRESSURE_VARIABLE
    scale : float
        - gaussian noise의 scale factor (wipe_out=False 일 때 사용)
    perturb_timestep : list[int]
        - perturb를 적용할 time index 리스트 (기본값: [0,1])
    num_points : int
        - 전체 (표면+대기압) 격자의 10%에 해당하는 개수 (기본값: 23568048)
    wipe_out : bool
        - True이면 mean 값으로 치환, False이면 (scale * std * Normal(0,1))을 더함
    region : dict
        - 특정 영역에서만 perturb를 주고 싶을 때 사용.
        - 예: region = {'lat': (lat_min, lat_max), 'lon': (lon_min, lon_max)}
        - None이면 전 지구 영역을 대상으로 perturb를 적용.
    """
    NONNEGATIVE_VARS = {"total_precipitation_6hr", "specific_humidity"}

    # ---- 사전 세팅
    perturbed = original.copy()

    # lat/lon 좌표값
    lat_vals = perturbed["lat"].values
    lon_vals = perturbed["lon"].values

    batch_size = perturbed.sizes.get('batch', 1)
    time_size = perturbed.sizes.get('time', 1)

    # region이 주어졌다면, 그 영역 내의 lat/lon 인덱스만 추출
    if region is not None:
        # region['lat'] = (lat_min, lat_max), region['lon'] = (lon_min, lon_max)
        lat_min, lat_max = region['lat']
        lon_min, lon_max = region['lon']

        # 해당 영역에 속하는 lat/lon 인덱스 마스킹
        lat_mask = (lat_vals >= lat_min) & (lat_vals <= lat_max)
        lon_mask = (lon_vals >= lon_min) & (lon_vals <= lon_max)

        # region 내 lat, lon 인덱스
        lat_idx_region = np.where(lat_mask)[0]  # 예: [10, 11, 12, ...] 형태
        lon_idx_region = np.where(lon_mask)[0]

        region_lat_size = len(lat_idx_region)
        region_lon_size = len(lon_idx_region)
    else:
        # region이 None이면 전 영역 사용
        lat_idx_region = np.arange(perturbed.sizes['lat'])
        lon_idx_region = np.arange(perturbed.sizes['lon'])
        region_lat_size = perturbed.sizes['lat']
        region_lon_size = perturbed.sizes['lon']

    # level 크기 (level 없는 경우 1로 처리)
    level_size = perturbed.sizes['level'] if 'level' in perturbed.sizes else 1

    # surface/pressure 변수 구분
    surface_vars = []
    pressure_vars = []
    for var in variables:
        if 'level' in perturbed[var].dims:
            pressure_vars.append(var)
        else:
            surface_vars.append(var)

    # (1) region 내부에서 변수별 격자 개수 계산
    #     surface: region_lat_size * region_lon_size
    #     pressure: level_size * region_lat_size * region_lon_size
    surf_grid = region_lat_size * region_lon_size
    press_grid = level_size * region_lat_size * region_lon_size

    var_sizes = []
    for var in surface_vars:
        var_sizes.append(surf_grid)
    for var in pressure_vars:
        var_sizes.append(press_grid)

    # jnp.array로 변환
    var_sizes_j = jnp.array(var_sizes)  # shape=(len(variables),)

    # var_offsets: 누적합 (0, var_sizes[0], var_sizes[0]+var_sizes[1], ...)
    var_offsets_j = jnp.cumsum(var_sizes_j)  # shape=(len(variables),)
    var_offsets_j = jnp.concatenate([jnp.array([0]), var_offsets_j])  # shape=(len(variables)+1,)

    # (2) JAX로 데이터를 옮겨놓기
    original_data = {}
    for var in variables:
        original_data[var] = device_put(jnp.array(perturbed[var].values))

    # 표준편차/평균 데이터셋 로드 (사용자 환경에 맞게 경로 수정)
    with xr.open_dataset("/geodata2/S2S/DL/GC_input/stat/stats_stddev_by_level.nc") as std_ds, \
         xr.open_dataset("/geodata2/S2S/DL/GC_input/stat/stats_mean_by_level.nc") as mean_ds:

        std_data = {}
        mean_data = {}
        for var in variables:
            std_data[var] = device_put(jnp.array(std_ds[var].values))
            mean_data[var] = device_put(jnp.array(mean_ds[var].values))

        # 랜덤 시드
        key = random.PRNGKey(42)

        # (3) time loop
        for t_idx in perturb_timestep:
            if t_idx >= time_size:
                continue  # time 범위 넘어가면 skip

            # batch loop
            for b_idx in range(batch_size):
                # 3-1) num_points개를 region 내 ( var_sizes 합 )에서 무작위로 뽑기
                total_points_region = var_offsets_j[-1]  # region 내 모든 변수의 격자 합
                key, subkey = random.split(key)
                rand_indices = random.choice(
                    subkey,
                    total_points_region,
                    shape=(num_points,),
                    replace=True  # region 범위 내에서 중복 허용
                )

                # 3-2) 각 랜덤 인덱스가 어떤 변수인지 searchsorted로 판별
                var_id = jnp.searchsorted(var_offsets_j, rand_indices, side='right') - 1
                local_index = rand_indices - var_offsets_j[var_id]

                # 3-3) 변수별로 모아서 perturb
                for i, var in enumerate(surface_vars + pressure_vars):
                    mask = (var_id == i)
                    count_i = jnp.sum(mask)
                    if count_i == 0:
                        continue

                    key, subkey = random.split(key)
                    normal_dist = random.normal(subkey, shape=(count_i,))

                    if var in surface_vars:
                        # unravel
                        local_index_surf = local_index[mask]
                        lat_idx_local, lon_idx_local = jnp.divmod(local_index_surf, region_lon_size)

                        # 실제 global lat/lon 인덱스로 변환
                        lat_idx_global = jnp.array(lat_idx_region)[lat_idx_local]
                        lon_idx_global = jnp.array(lon_idx_region)[lon_idx_local]

                        if wipe_out:
                            original_data[var] = original_data[var].at[
                                b_idx, t_idx, lat_idx_global, lon_idx_global
                            ].set(mean_data[var])
                        else:
                            # lat_rad = jnp.deg2rad(jnp.array(lat_vals)[lat_idx_global])  # degree -> rad
                            # cos_lat = jnp.cos(lat_rad)

                            noise = scale * std_data[var] * normal_dist # * cos_lat

                            original_data[var] = original_data[var].at[
                                b_idx, t_idx, lat_idx_global, lon_idx_global
                            ].add(noise)

                        if var in NONNEGATIVE_VARS:
                            old_vals = original_data[var][b_idx, t_idx, lat_idx_global, lon_idx_global]
                            clipped_vals = jnp.clip(old_vals, 0.0, None)  # 최소 0
                            original_data[var] = original_data[var].at[
                                b_idx, t_idx, lat_idx_global, lon_idx_global
                            ].set(clipped_vals)
                            
                    else:
                        # pressure var
                        local_index_press = local_index[mask]
                        lvl_idx_local = local_index_press // (region_lat_size * region_lon_size)
                        latlon_local = local_index_press % (region_lat_size * region_lon_size)
                        lat_idx_local, lon_idx_local = jnp.divmod(latlon_local, region_lon_size)

                        # 실제 global lat/lon 인덱스로 변환
                        lat_idx_global = jnp.array(lat_idx_region)[lat_idx_local]
                        lon_idx_global = jnp.array(lon_idx_region)[lon_idx_local]

                        if wipe_out:
                            original_data[var] = original_data[var].at[
                                b_idx, t_idx, lvl_idx_local, lat_idx_global, lon_idx_global
                            ].set(mean_data[var][lvl_idx_local])
                        else:
                            # lat_rad = jnp.deg2rad(jnp.array(lat_vals)[lat_idx_global])
                            # cos_lat = jnp.cos(lat_rad)

                            noise = scale * std_data[var][lvl_idx_local] * normal_dist # * cos_lat

                            original_data[var] = original_data[var].at[
                                b_idx, t_idx, lvl_idx_local, lat_idx_global, lon_idx_global
                            ].add(noise)

                        if var in NONNEGATIVE_VARS:
                            old_vals = original_data[var][b_idx, t_idx, lvl_idx_local, lat_idx_global, lon_idx_global]
                            clipped_vals = jnp.clip(old_vals, 0.0, None)
                            original_data[var] = original_data[var].at[
                                b_idx, t_idx, lvl_idx_local, lat_idx_global, lon_idx_global
                            ].set(clipped_vals)

        # (4) perturb된 결과를 numpy array로 되돌려서 xarray에 붙여넣기
        for var in variables:
            perturbed[var].values = np.array(original_data[var])

        # 혹시 'perturbation' 변수가 있었다면 제거
        if 'perturbation' in perturbed.data_vars:
            perturbed = perturbed.drop_vars('perturbation')

    return perturbed


def add_shuffle_perturbation(original: xr.Dataset, 
                                variables: list[str] = SURFACE_VARIABLE + PRESSURE_VARIABLE, 
                                scale: float = 1, 
                                perturb_timestep: list = [0, 1],
                                num_points: int = 23568048, 
                                wipe_out: bool = True):
    """
    original : xr.Dataset
        - (batch, time, lat, lon, [level]) 구조의 원본 데이터셋
    variables : list[str]
        - perturb를 적용할 변수 목록
        - 예) SURFACE_VARIABLE + PRESSURE_VARIABLE
    scale : float
        - gaussian noise의 scale factor (wipe_out=False 일 때 사용)
    perturb_timestep : list[int]
        - perturb를 적용할 time index 리스트 (기본값: [0,1])
    num_points : int
        - 전체 (표면+대기압) 격자의 10%에 해당하는 개수 (기본값: 23568048)
    wipe_out : bool
        - True이면 mean 값으로 치환, False이면 (scale * std * Normal(0,1))을 더함
    """
    # ---- 사전 세팅
    perturbed = original.copy()

    # lat/lon/level 크기
    lat_size = perturbed.sizes['lat']
    lon_size = perturbed.sizes['lon']
    level_size = perturbed.sizes['level'] if 'level' in perturbed.sizes else 1
    batch_size = perturbed.sizes['batch'] if 'batch' in perturbed.sizes else 1
    time_size = perturbed.sizes['time'] if 'time' in perturbed.sizes else 1

    # SURFACE_VARIABLE, PRESSURE_VARIABLE 식별
    # (여기서는 사용자가 직접 변수 리스트를 건네주므로, 실제 코드에서는
    #  surface인지, pressure인지 확인할 때 변수를 보고 판단해야 함)
    #  예: 5개 surface 변수, 6개 pressure 변수가 들어온다고 가정
    #  (만약 SURFACE_VARIABLE, PRESSURE_VARIABLE을 직접 분류할 필요가 있다면
    #   별도 로직을 추가)
    surface_vars = []
    pressure_vars = []
    for var in variables:
        if 'level' in perturbed[var].dims:
            pressure_vars.append(var)
        else:
            surface_vars.append(var)

    # (1) 변수별 격자 개수 계산
    # surface: lat_size * lon_size
    # pressure: level_size * lat_size * lon_size
    surf_grid = lat_size * lon_size
    press_grid = level_size * lat_size * lon_size

    # var_sizes: 각 변수(=DataArray)가 몇 개의 격자를 갖고 있는지
    var_sizes = []
    for var in surface_vars:
        var_sizes.append(surf_grid)
    for var in pressure_vars:
        var_sizes.append(press_grid)

    # jnp.array로 변환
    var_sizes_j = jnp.array(var_sizes)  # shape=(len(variables),)

    # var_offsets: 누적합 (0, var_sizes[0], var_sizes[0]+var_sizes[1], ...)
    var_offsets_j = jnp.cumsum(var_sizes_j)  # shape=(len(variables),)
    # searchsorted 할 때 편의를 위해 앞에 0을 하나 더 붙이거나,
    # side='right' 옵션을 잘 쓸 수도 있음.
    # 여기서는 side='right'을 쓰기 위해, offsets 앞에 0을 붙인 버전 사용
    var_offsets_j = jnp.concatenate([jnp.array([0]), var_offsets_j])  # shape=(len(variables)+1,)

    # (2) JAX로 데이터를 옮겨놓기
    original_data = {}
    for var in variables:
        original_data[var] = device_put(jnp.array(perturbed[var].values))

    # 표준편차/평균 데이터셋 로드 (사용자 환경에 맞게 경로 수정)
    with xr.open_dataset("/geodata2/S2S/DL/GC_input/stat/stats_stddev_by_level.nc") as std_ds, \
         xr.open_dataset("/geodata2/S2S/DL/GC_input/stat/stats_mean_by_level.nc") as mean_ds:

        std_data = {}
        mean_data = {}
        for var in variables:
            std_data[var] = device_put(jnp.array(std_ds[var].values))
            mean_data[var] = device_put(jnp.array(mean_ds[var].values))

        # 랜덤 시드
        key = random.PRNGKey(42)

        # (3) time loop
        for t_idx in perturb_timestep:
            if t_idx >= time_size:
                continue  # time 범위 넘어가면 skip

            # batch loop (예: batch=1이면 사실상 1번만)
            for b_idx in range(batch_size):
                # 3-1) num_points개를 전체( var_sizes 합 )에서 무작위로 뽑기
                total_points = var_offsets_j[-1]  # 모든 변수의 격자 총합
                key, subkey = random.split(key)
                rand_indices = random.choice(
                    subkey,
                    total_points,        # [0..total_points-1] 중에서
                    shape=(num_points,), # num_points 개를
                    replace=True        # 중복 없이 뽑음
                )

                # 3-2) 각 랜덤 인덱스가 어떤 변수인지 searchsorted로 판별
                # var_id: 0 ~ (len(variables)-1)
                # var_offsets_j: 누적합, shape=(len(variables)+1,)
                # 예: var_offsets_j = [0, surf_grid, 2*surf_grid, 2*surf_grid + press_grid, ...]
                # idx ∈ [var_offsets_j[v], var_offsets_j[v+1]) 이면 var_id = v
                var_id = jnp.searchsorted(var_offsets_j, rand_indices, side='right') - 1
                # local_index = rand_indices - var_offsets_j[var_id]
                local_index = rand_indices - var_offsets_j[var_id]

                # 3-3) 변수별로 모아서 perturb
                #      예: var_id == 0 인 것, var_id == 1 인 것 등등
                for i, var in enumerate(surface_vars + pressure_vars):
                    # 현재 var_id=i 인 random index들만 뽑는다
                    mask = (var_id == i)
                    # 그 중 실제 개수
                    count_i = jnp.sum(mask)
                    if count_i == 0:
                        continue

                    # 해당 var가 surface인지 pressure인지 구분
                    if var in surface_vars:
                        # local_index_surf: shape=(count_i,)
                        local_index_surf = local_index[mask]
                        # unravel: lat, lon
                        lat_idx, lon_idx = jnp.divmod(local_index_surf, lon_size)

                        # 가우시안 noise
                        key, subkey = random.split(key)
                        normal_dist = random.normal(subkey, shape=(count_i,))

                        if wipe_out:
                            # set mean
                            # mean_data[var]는 보통 (lat, lon) 또는 (level, lat, lon) 형식이지만
                            # 통계파일이 어떻게 만들어졌느냐에 따라 다름.
                            # 여기서는 'surface'에 level이 없다고 가정 → mean_data[var]는 2D or 1D
                            # 만약 mean_data[var]가 (lat, lon)이면, 해당 lat_idx, lon_idx 위치를 뽑아야 함.
                            # (아래에서는 "scalar or 2D 모두 가능"하다고 보고 단순히 scalar라 가정)
                            original_data[var] = original_data[var].at[
                                b_idx, t_idx, lat_idx, lon_idx
                            ].set(mean_data[var])  # 만약 mean_data[var]가 scalar라면 이렇게
                        else:
                            # add noise
                            original_data[var] = original_data[var].at[
                                b_idx, t_idx, lat_idx, lon_idx
                            ].add(scale * std_data[var] * normal_dist)

                    else:
                        # pressure var
                        local_index_press = local_index[mask]
                        # unravel: level, lat, lon
                        lvl_idx = local_index_press // (lat_size * lon_size)
                        latlon_local = local_index_press % (lat_size * lon_size)
                        lat_idx, lon_idx = jnp.divmod(latlon_local, lon_size)

                        # 가우시안 noise
                        key, subkey = random.split(key)
                        normal_dist = random.normal(subkey, shape=(count_i,))

                        if wipe_out:
                            # set mean
                            # mean_data[var]가 (level, lat, lon)인지, (level,) 인지 등
                            # 실제 통계파일 구조에 맞게 처리
                            original_data[var] = original_data[var].at[
                                b_idx, t_idx, lvl_idx, lat_idx, lon_idx
                            ].set(mean_data[var][lvl_idx])
                        else:
                            original_data[var] = original_data[var].at[
                                b_idx, t_idx, lvl_idx, lat_idx, lon_idx
                            ].add(scale * std_data[var][lvl_idx] * normal_dist)

        # (4) perturb된 결과를 np.array로 되돌려서 xarray에 붙여넣기
        for var in variables:
            perturbed[var].values = np.array(original_data[var])

        # 혹시 'perturbation' 변수가 있었다면 제거
        if 'perturbation' in perturbed.data_vars:
            perturbed = perturbed.drop_vars('perturbation')

    return perturbed


def add_region_specific_perturbation(original: xr.Dataset, 
                                     variables: list[str], 
                                     scale: float, 
                                     perturb_timestep: list = [0, 1],
                                     region_list: list[str] = [],
                                     #if wipe_out is True, scale is the value to set
                                     wipe_out: bool = False): 
    # Remove 'batch' dimension if present
    if "batch" in original.dims:
        original = original.squeeze("batch")

    # Move original dataset to GPU using JAX
    original_data = {var: device_put(jnp.array(original[var].values)) for var in variables}
    lat_values = device_put(jnp.array(original.lat.values))
    lon_values = device_put(jnp.array(original.lon.values))

    # Handle longitude values, making sure they are in the correct range
    if lon_values.min() < 0:
        lon_values = (lon_values + 360) % 360  # Convert range from [-180, 180] to [0, 360]

    with xr.open_dataset("/geodata2/S2S/DL/GC_input/stat/stats_stddev_by_level.nc") as std:
        std_data = {}
        for var in variables:
            if var in std:
                std_data[var] = device_put(jnp.array(std[var].values))
            else:
                # If the variable is not in std, set a default scalar value
                std_data[var] = device_put(jnp.array(1.0))  # Or any default value you deem appropriate

        perturbed = original.copy()
        
        # Initialize lists to collect indices
        selected_lat_indices = []
        selected_lon_indices = []
        
        for region in region_list:
            if region in REGION_BOUNDARIES:
                region_bounds = REGION_BOUNDARIES[region]
                
                # Latitude indices
                lat_min, lat_max = region_bounds["lat"]
                lat_mask = (lat_values >= lat_min) & (lat_values <= lat_max)
                lat_indices = jnp.where(lat_mask)[0]
                if lat_indices.size == 0:
                    continue  # Skip if no matching latitudes are found
                
                # Longitude indices
                if "lon" in region_bounds:
                    lon_min, lon_max = region_bounds["lon"]
                    if lon_min < 0:
                        lon_min = (lon_min + 360) % 360
                    if lon_max < 0:
                        lon_max = (lon_max + 360) % 360
                        
                    if lon_min < lon_max:
                        lon_mask = (lon_values >= lon_min) & (lon_values <= lon_max)
                    else:  # Handle wrapping around the 0-degree longitude
                        lon_mask = (lon_values >= lon_min) | (lon_values <= lon_max)
                else:
                    lon_mask = jnp.ones_like(lon_values, dtype=bool)  # Select all longitudes
                
                lon_indices = jnp.where(lon_mask)[0]
                if lon_indices.size == 0:
                    continue  # Skip if no matching longitudes are found
                
                # Create a grid of lat/lon indices
                lat_grid, lon_grid = jnp.meshgrid(lat_indices, lon_indices, indexing='ij')
                selected_lat_indices.append(lat_grid.flatten())
                selected_lon_indices.append(lon_grid.flatten())
        
        if not selected_lat_indices or not selected_lon_indices:
            # If no indices are selected, return the original dataset without modifications
            return perturbed

        # Concatenate indices from all regions
        selected_lat_indices = jnp.concatenate(selected_lat_indices)
        selected_lon_indices = jnp.concatenate(selected_lon_indices)

        # Create a normal distribution for the selected points using JAX
        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        normal_dist = random.normal(subkey, shape=(selected_lat_indices.size,))

        # Convert perturb_timestep to JAX array for efficient broadcasting
        perturb_timesteps = jnp.array(perturb_timestep)

        # Apply perturbations using matrix operations
        for var in variables:
            std_var = std_data[var]
            data_var = original_data[var]

            if 'level' in original[var].dims:
                for level_idx in range(original.level.size):
                    if wipe_out:
                        data_var = data_var.at[
                            perturb_timesteps[:, None], level_idx, selected_lat_indices[None, :], selected_lon_indices[None, :]
                        ].set(scale)
                    else:
                        if std_var.ndim == 3:
                            std_values = std_var[level_idx, selected_lat_indices, selected_lon_indices]
                        elif std_var.ndim == 1:
                            std_values = std_var[level_idx]
                        elif std_var.ndim == 0:
                            std_values = std_var
                        else:
                            raise ValueError(f"Unexpected dimensions for std_data[{var}]: {std_var.shape}")
                        perturbation = scale * normal_dist[None, :] * std_values
                        data_var = data_var.at[
                            perturb_timesteps[:, None], level_idx, selected_lat_indices[None, :], selected_lon_indices[None, :]
                        ].add(perturbation)
                original_data[var] = data_var
            else:
                if wipe_out:
                    data_var = data_var.at[
                        perturb_timesteps[:, None], selected_lat_indices[None, :], selected_lon_indices[None, :]
                    ].set(scale)
                else:
                    if std_var.ndim == 2:
                        std_values = std_var[selected_lat_indices, selected_lon_indices]
                    elif std_var.ndim == 0:
                        std_values = std_var
                    else:
                        raise ValueError(f"Unexpected dimensions for std_data[{var}]: {std_var.shape}")
                    perturbation = scale * normal_dist[None, :] * std_values
                    data_var = data_var.at[
                        perturb_timesteps[:, None], selected_lat_indices[None, :], selected_lon_indices[None, :]
                    ].add(perturbation)
                original_data[var] = data_var

        # Update the perturbed dataset with the modified values
        for var in variables:
            perturbed[var].values = np.array(original_data[var])
                
        # Drop 'perturbation' variable if it exists
        if 'perturbation' in perturbed.data_vars:
            perturbed = perturbed.drop_vars('perturbation')

    # Transpose and expand dims as needed
    perturbed = perturbed.transpose('time', 'lat', 'lon', 'level').expand_dims('batch')
    if 'batch' not in perturbed['datetime'].dims:
        perturbed['datetime'] = perturbed['datetime'].expand_dims({'batch': [0]}, axis=0)

    # Squeeze 'batch' dimension for specific variables if needed
    for var in ['geopotential_at_surface', 'land_sea_mask']:
        if var in perturbed and 'batch' in perturbed[var].dims:
            perturbed[var] = perturbed[var].squeeze('batch')

    return perturbed



def add_region_perturbation(original: xr.Dataset, 
                            variables: list[str], 
                            scale: float, 
                            perturb_timestep: list = [0, 1],
                            num_points: int = 100,
                            wipe_out: bool = False):
    # Remove 'batch' dimension if present
    if "batch" in original.dims:
        original = original.squeeze("batch")

    # Move original dataset to GPU using JAX
    original_data = {var: device_put(jnp.array(original[var].values)) for var in variables}
    lat_values = device_put(jnp.array(original.lat.values))
    lon_values = device_put(jnp.array(original.lon.values))

    with xr.open_dataset("/geodata2/S2S/DL/GC_input/stat/stats_stddev_by_level.nc") as std, xr.open_dataset("/geodata2/S2S/DL/GC_input/stat/stats_mean_by_level.nc") as mean:  
        std_data = {var: device_put(jnp.array(std[var].values)) for var in variables}
        mean_data = {var: device_put(jnp.array(mean[var].values)) for var in variables}
        perturbed = original.copy()
        
        # Generate random lat/lon pairs to perturb from the entire grid using JAX
        key = random.PRNGKey(0)
        lat_lon_choices = random.choice(key, lat_values.size * lon_values.size, shape=(num_points,), replace=False)
        lat_indices, lon_indices = jnp.unravel_index(lat_lon_choices, (lat_values.size, lon_values.size))
        selected_lat_indices = lat_indices
        selected_lon_indices = lon_indices

        # Create a normal distribution for the selected points using JAX
        key, subkey = random.split(key)
        normal_dist = random.normal(subkey, shape=(num_points,))

        # Convert perturb_timestep to JAX array for efficient broadcasting
        perturb_timesteps = jnp.array(perturb_timestep)

        # Apply perturbations using matrix operations
        for var in variables:
            if 'level' in original[var].dims:
                for level_idx in range(original.level.size):
                    if wipe_out:
                        # Overwrite with mean value
                        original_data[var] = original_data[var].at[
                            perturb_timesteps[:, None], level_idx, selected_lat_indices[None, :], selected_lon_indices[None, :]
                        ].set(mean_data[var][level_idx])
                    else:
                        # Broadcast perturbation over all specified timesteps and selected lat/lon points
                        original_data[var] = original_data[var].at[
                            perturb_timesteps[:, None], level_idx, selected_lat_indices[None, :], selected_lon_indices[None, :]
                        ].add(scale * normal_dist[None, :] * std_data[var][level_idx])
            else:
                if wipe_out:
                    # Overwrite with mean value
                    original_data[var] = original_data[var].at[
                        perturb_timesteps[:, None], selected_lat_indices[None, :], selected_lon_indices[None, :]
                    ].set(mean_data[var])
                else:
                    # Apply perturbation without level dimension
                    original_data[var] = original_data[var].at[
                        perturb_timesteps[:, None], selected_lat_indices[None, :], selected_lon_indices[None, :]
                    ].add(scale * normal_dist[None, :] * std_data[var])

        # Update the perturbed dataset with the modified values
        for var in variables:
            perturbed[var].values = np.array(original_data[var])
                
        # Drop 'perturbation' variable if it exists
        if 'perturbation' in list(perturbed.data_vars):
            perturbed = perturbed.drop_vars('perturbation')

    # Transpose and expand dims as needed
    perturbed = perturbed.transpose('time', 'lat', 'lon', 'level').expand_dims('batch')
    if 'batch' not in perturbed['datetime'].dims:
        perturbed['datetime'] = perturbed['datetime'].expand_dims({'batch': [0]}, axis=0)

    # Squeeze 'batch' dimension for specific variables if needed
    for var in ['geopotential_at_surface', 'land_sea_mask']:
        if var in perturbed and 'batch' in perturbed[var].dims:
            perturbed[var] = perturbed[var].squeeze('batch')

    return perturbed


# Add Gaussian perturbation to the original dataset.
def add_Gaussian_perturbation(original: xr.Dataset, 
                     variables: list[str], 
                     scale: float, 
                     perturb_timestep: list = [0, 1]):
   
   
    if "batch" in original.dims:
        original = original.squeeze("batch")

    with xr.open_dataset("/geodata2/S2S/DL/GC_input/stat/stats_stddev_by_level.ncc") as std: 
        perturbed = original.copy()
        
        for i in perturb_timestep:
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
                if 'level' in original[var].dims:
                    for level in original.level:
                        perturbed.isel(time=i, level=level)[var] = original[var].isel(time=i, level=level) + scale * (normal_dist * std[var].isel(level=level))
                else:                           
                    perturbed.isel(time=i)[var] = original[var].isel(time=i) + scale * (normal_dist * std[var])
                
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



def transform_dataset(dataset, d_t = "6h"):
    """
        Transform the dataset to the format such as variable names and coordinates required by the model.
        Referenced at data_concat.ipynb
    """
    # 1. batch 차원 추가
    if 'batch' not in dataset.dims:
        dataset = dataset.expand_dims(dim={'batch': [0]})
    
    tp = f"total_precipitation_{d_t}r"

    # 2. 변수 이름 매핑
    name_mapping = {
        'u10': '10m_u_component_of_wind',
        'v10': '10m_v_component_of_wind',
        't2m': '2m_temperature',
        'msl': 'mean_sea_level_pressure',
        'tp': tp,
        'sst': 'sea_surface_temperature',
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
    dataset.coords['datetime'] = ('time', pd.date_range(start=start_time, periods=len(dataset.time), freq=d_t))
    dataset['datetime'] = dataset['datetime'].expand_dims({'batch': [0]}, axis=0)

    for var in ['geopotential_at_surface', 'land_sea_mask']:
        if var in dataset:
            dataset[var] = dataset[var].isel(time=0, drop=True)
            if 'batch' in dataset[var].dims:
                dataset[var] = dataset[var].squeeze('batch')
    
    return dataset.reindex(lat=dataset.lat[::-1])



def create_forcing_dataset(time_steps, 
                           resolution, 
                           start_time,
                           d_t = 6):
    lon = np.arange(0.0, 360.0, resolution, dtype=np.float32)
    
    start_datetime = pd.to_datetime(start_time) + pd.Timedelta(hours=12)
    datetime = pd.date_range(start=start_datetime, 
                             periods=time_steps, 
                             freq=f"{d_t}h")
    
    
    time = pd.timedelta_range(start=pd.Timedelta(hours=d_t), 
                              periods=time_steps, 
                              freq=f"{d_t}h")
        
    variables = ['year_progress_sin',
                 'year_progress_cos',
                 'day_progress_sin',
                 'day_progress_cos']
    
    if d_t == 6:
        lat = np.arange(-90.0, 90.0 + resolution/2, resolution, dtype=np.float32)
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

        variables.append('toa_incident_solar_radiation')
        
        data_utils.add_tisr_var(ds)
    else:
        ds = xr.Dataset(
                coords={
                    'lon': ('lon', lon),
                    'datetime': ('datetime', datetime),
                    'time': ('time', time)
                    }
                )
    
    ds.lon.attrs['long_name'] = 'longitude'
    ds.lon.attrs['units'] = 'degrees_east'

    data_utils.add_derived_vars(ds)
    
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



def create_target_dataset(
        time_steps, 
        resolution, 
        pressure_levels, 
        d_t = 6):
    
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
    time = pd.timedelta_range(start=pd.Timedelta(hours=d_t), 
                              periods=time_steps, 
                              freq=f"{d_t}h")
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

    surface_vars = ['2m_temperature', 
                        'mean_sea_level_pressure', 
                        '10m_v_component_of_wind', 
                        '10m_u_component_of_wind']
    # Add data variables filled with NaN
    if d_t == 6:
        surface_vars.append('total_precipitation_6hr')
    else:
        surface_vars.append('total_precipitation_12hr')
        surface_vars.append('sea_surface_temperature')
    
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
