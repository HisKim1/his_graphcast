import xarray as xr
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# fine
SURFACE_VARIABLE = ['u_wind',
                 'v_wind',
                 # 'total_precip', # skip this
                 'sea_level_pressure',
                 't2m'
                 ]

PRESSURE_VARIALBE = ['temperature', # saved monthly
                     'u_wind',# saved monthly / till 1987_10
                     'v_wind', # saved monthly / till 1989_09
                     'specific_humidity', # saved monthly / till 1989_10
                     'geopotential' # missing 
                     ]


parser = argparse.ArgumentParser(description='run std calculator')
parser.add_argument('--target', type=str, choices=SURFACE_VARIABLE+PRESSURE_VARIALBE, required=True)
parser.add_argument('--is_pressure', type=int, required=True)

target = parser.parse_args().target

is_pressure = parser.parse_args().is_pressure
start_year = 1979
end_year = 2017
input_dir = '/camdata2/ERA5/hourly/'
output_dir = '../testdata/stats/'
num_workers = 40

if bool(is_pressure):
    input_dir += 'pressure/'
    output_dir += 'pressure/'
input_dir += target
print("Input directory: ", input_dir)

if target in ['specific_humidity', 'v_wind'] and bool(is_pressure):
    end_year = 1988
elif target in ['u_wind'] and bool(is_pressure):
    end_year = 1988


os.makedirs(output_dir, exist_ok=True)


def process_year(year):
    filename = f"{year}.nc"
    filepath = os.path.join(input_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Warning: File for year {year} not found.")
        return None
    
    ds = xr.open_dataset(filepath)
    
    results = {}
    for hour in ['00', '06', '12', '18']:
        hourly_data = ds.sel(time=ds.time.dt.hour == int(hour))
        results[hour] = hourly_data
    
    return results


def try_this():
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_year = {executor.submit(process_year, year): year for year in range(start_year, end_year + 1)}
        
        datasets = {hour: [] for hour in ['00', '06', '12', '18']}
        for future in as_completed(future_to_year):
            year = future_to_year[future]
            try:
                data = future.result()
                if data:
                    for hour in ['00', '06', '12', '18']:
                        datasets[hour].append(data[hour])
            except Exception as exc:
                print(f'Year {year} generated an exception: {exc}')

    for hour in ['00', '06', '12', '18']:
        print(f"Combining data for {hour}:00")
        combined_data = xr.concat(datasets[hour], dim='time')
        output_file = os.path.join(output_dir, f'40yr_{hour}h.nc')
        print(f"Saving {output_file}")
        combined_data.to_netcdf(output_file)

    print("Processing complete.")

try_this()

name_mapping = {
    't2m': '2m_temperature',
    't': 'temperature',
    'u': 'u10',
    'v': 'v10',
    'z': 'geopotential',
    'q': 'specific_humidity',
    'longitude': 'lon',
    'latitude': 'lat'
}

for time in ["00", "06", "12", "18"]:
    hxx = xr.open_dataset(f'{output_dir}/40yr_{time}h_std_daily.nc')
    hxx = hxx.rename({k: v for k, v in name_mapping.items() 
                      if k in hxx.variables or k in hxx.coords})
    hxx.to_netcdf(f'{output_dir}/40yr_{time}h_std_daily.nc')

import glob
files = sorted(glob.glob(f'{output_dir}/40yr_*h_std_daily.nc'))
print(files)
std = xr.open_mfdataset(files, combine='nested', concat_dim='hour')
std.compute().to_netcdf(f'{output_dir}/40yr_std_daily.nc')
