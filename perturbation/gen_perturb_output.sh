#!/bin/bash

model="original"
eval_steps=28
input_files=$(ls /geodata2/S2S/DL/GC_input/2021-06-21/ERA5_*)
input_dir="/geodata2/S2S/DL/GC_input/2021-06-21/"
output_dir="/geodata2/S2S/DL/GC_output/2021-06-21"

echo "Input files: ${input_files}"
echo "============="
tac filelist.txt | while read input_file
do
    echo ">>>> $input_dir$input_file"
    # std_value=$(basename "$input_file" | grep -oP '(?<=_)\d+(\.\d+)?(?=std\.nc)')
    # echo "std: $std_value"
    # output_file="${output_dir}/GC_${std_value}std.nc"
    filename=$(basename "$input_file")
    new_filename="${filename/ERA5_/GC_}"
    output_file="${output_dir}/${new_filename}"

    if [ -f "$output_file" ]; then
        echo "Output file exists: $output_file"
        continue
    fi

    python ~/his_graphcast/GC_run.py --input "$input_dir$input_file" --output "$output_file" --model "$model" --eval_steps "$eval_steps"
    chmod 777 "$output_file"
    echo "============="
done 

# python ~/graphcast/GC_run.py --input /geodata2/S2S/DL/GC_input/2021-06-21/ERA5_input.nc --output /geodata2/S2S/DL/GC_output/2021-06-21/GC_output.nc --model original --eval_steps 28