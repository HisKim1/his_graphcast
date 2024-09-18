#!/bin/bash

# Define the variables
SURFACE_VARIABLES=("u_wind" "v_wind" "total_precip" "sea_level_pressure" "t2m")
PRESSURE_VARIABLES=("temperature" "u_wind" "v_wind", "specific_humidity")

# need "geopotential" /   missing

# Function to run the Python script
run_std_calculator() {
    local variable=$1
    local is_pressure=$2
    echo "Processing $variable..."
    python std_calculator2.py --target "$variable" --is_pressure "$is_pressure"
    
    if [ $? -ne 0 ]; then
        echo "Error processing $variable"
    else
        echo "Finished processing $variable"
    fi
    echo "------------------------"
}

# Process surface variables
echo "Processing Surface Variables"
for var in "${SURFACE_VARIABLES[@]}"; do
    run_std_calculator "$var" 0
done

# Process pressure variables
# echo "Processing Pressure Variables"
# for var in "${PRESSURE_VARIABLES[@]}"; do
#     run_std_calculator "$var" 1
# done

echo "All variables processed."