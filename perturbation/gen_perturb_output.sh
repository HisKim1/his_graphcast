#!/bin/bash

model="original"
eval_steps=40
input_files=$(ls ~/graphcast/testdata/2021-06-21/ERA5_4*)
output_dir="~/graphcast/testdata/2021-06-21"

echo "Input files: ${input_files}"
echo "============="
for input_file in $input_files; do
    echo ">>>> $input_file..."
    std_value=$(basename "$input_file" | grep -oP '(?<=_)\d+(\.\d+)?(?=std\.nc)')
    echo "std: $std_value"
    output_file="${output_dir}/GC_4var_${std_value}std.nc"

    
    echo "<<<< $output_file"
    python ~/graphcast/GC_run.py --input "$input_file" --output "$output_file" --model "$model" --eval_steps "$eval_steps"
    
    echo "============="
done

python ~/graphcast/GC_run.py --input ~/graphcast/testdata/2021-06-21/ERA5_input.nc --output ~/graphcast/testdata/2021-06-21/GC_output.nc --model original --eval_steps 40