import xarray as xr
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from lib import his_utils
import os
import itertools

scales = [0.01, 0.05, 0.1, 0.5]

ten_persent = 103680

variables = ['10m_u_component_of_wind',
 '10m_v_component_of_wind',
 '2m_temperature',
 'mean_sea_level_pressure',
 'total_precipitation_6hr',
 'geopotential',
 'specific_humidity',
 'temperature',
 'u_component_of_wind',
 'v_component_of_wind',
 'vertical_velocity']

def binary_string(combo):
    # Convert the selected combination to a binary string
    return ''.join(['1' if var in combo else '0' for var in variables])

# Step 1: Select 1 
one_two_variable_combinations = []
for r in [1]:
    one_two_variable_combinations += list(itertools.combinations(variables, r))

# Step 2: Select all but 1 variable
all_but_one_variable_combinations = list(itertools.combinations(variables, len(variables) - 1))

# Step 3: Select all variables
all_variable_combination = [tuple(variables)]  # All variables selected

# Combine all cases
selected_combinations = one_two_variable_combinations + all_but_one_variable_combinations + all_variable_combination

# Loop through selected combinations
for combo in  [('2m_temperature',)]:
    combo_string = binary_string(combo)
    for scale in scales:
        for n in range(1, 11, 2):
            # Generate a descriptive filename
            wipeout_str = "scale"
            filename = f"ERA5_{combo_string}_{scale}_{n}p.nc"
            if os.path.exists(os.path.join('/geodata2/S2S/DL/GC_input/percent2', filename)):
                print(f"Skipping: {filename}")
                continue
            # Apply the perturbation to the dataset
            dataset = xr.open_dataset('/geodata2/S2S/DL/GC_input/2021-06-21/ERA5_input.nc')
            perturbed_dataset = his_utils.add_region_perturbation(
                dataset, list(combo), scale, perturb_timestep=[0, 1], num_points=ten_persent*n, wipe_out=False
            )
            # Save to a new compressed file
            output_path = os.path.join('/geodata2/S2S/DL/GC_input/percent2', filename)
            encoding = {var: {'zlib': True, 'complevel': 5} for var in perturbed_dataset.variables}
            perturbed_dataset.to_netcdf(output_path, encoding=encoding)
            print(f"Created: {filename}")
            dataset.close()
            perturbed_dataset.close()