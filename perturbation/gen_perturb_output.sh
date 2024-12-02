#!/bin/bash

model="original"
eval_steps=28
input_files=$(ls /data/GC_input/percent/ERA5_* | tee filelist.txt)
input_dir="/data/GC_input/percent/"
output_dir="/data/GC_output/percent"

echo ========================================================
cat filelist.txt | while read input_file
do
    echo ">>>> $input_file"
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

    # echo "Output file: $output_file"
    python ~/graphcast/GC_run.py --input "$input_file" --output "$output_file" --model "$model" --eval_steps "$eval_steps"
    chmod 777 "$output_file"
    echo "========================================================"
done 