import xarray as xr
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import numpy as np

SURFACE_VARIABLES = ['u_wind', 'v_wind', 'sea_level_pressure', 't2m']
PRESSURE_VARIABLES = ['temperature', 'u_wind', 'v_wind', 'specific_humidity', 'geopotential']

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run std calculator for ERA5 data')
    parser.add_argument('--target', type=str, choices=SURFACE_VARIABLES + PRESSURE_VARIABLES, required=True)
    parser.add_argument('--is_pressure', type=int, required=True)
    return parser.parse_args()

def setup_directories(args):
    base_input_dir = '/camdata2/ERA5/hourly/'
    base_output_dir = '../testdata/stats/'
    
    if bool(args.is_pressure):
        base_input_dir += 'pressure/'
        base_output_dir += 'pressure/'
    
    input_dir = os.path.join(base_input_dir, args.target)
    os.makedirs(base_output_dir, exist_ok=True)
    
    return input_dir, base_output_dir

def determine_year_range(args):
    start_year = 1979
    end_year = 2017
    
    if bool(args.is_pressure):
        if args.target in ['specific_humidity', 'v_wind']:
            end_year = 1988
        elif args.target == 'u_wind':
            end_year = 1988
    
    return start_year, end_year

def process_file(filepath, target_hours):
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return None
    
    with xr.open_dataset(filepath) as ds:
        results = {}
        for hour in target_hours:
            hourly_data = ds.sel(time=ds.time.dt.hour == int(hour))
            results[hour] = hourly_data[list(hourly_data.data_vars)[0]]  # Assuming single variable per file
    
    return results

def process_data(input_dir, start_year, end_year, target_hours, num_workers=40):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_year = {
            executor.submit(process_file, os.path.join(input_dir, f"{year}.nc"), target_hours): year
            for year in range(start_year, end_year + 1)
        }
        
        datasets = {hour: [] for hour in target_hours}
        for future in as_completed(future_to_year):
            year = future_to_year[future]
            try:
                data = future.result()
                if data:
                    for hour in target_hours:
                        datasets[hour].append(data[hour])
            except Exception as exc:
                print(f'Year {year} generated an exception: {exc}')
    
    return datasets

def calculate_std(datasets, target_hours):
    std_results = {}
    for hour in target_hours:
        print(f"Calculating std for {hour}:00")
        combined_data = xr.concat(datasets[hour], dim='time')
        std_results[hour] = combined_data.std(dim='time')
    return std_results

def save_final_result(std_results, output_dir, target_variable):
    print("Combining all hours and saving final result")
    combined_std = xr.concat([std_results[hour] for hour in std_results], dim='hour')
    combined_std = combined_std.assign_coords(hour=['00', '06', '12', '18'])
    
    output_file = os.path.join(output_dir, f'40yr_std_daily_{target_variable}.nc')
    print(f"Saving {output_file}")
    combined_std.to_netcdf(output_file)

def main():
    args = parse_arguments()
    input_dir, output_dir = setup_directories(args)
    start_year, end_year = determine_year_range(args)
    target_hours = ['00', '06', '12', '18']
    
    print(f"Processing {args.target} data from {start_year} to {end_year}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    datasets = process_data(input_dir, start_year, end_year, target_hours)
    std_results = calculate_std(datasets, target_hours)
    save_final_result(std_results, output_dir, args.target)
    
    print("Processing complete.")

if __name__ == "__main__":
    main()