#!/bin/bash

# 파일 경로 패턴 저장


model="original"
eval_steps=40

# input_files=$(ls /geodata2/S2S/DL/GC_input/proportional/ERA5_*_*_*.nc | tee filelist.txt)
input_dir="/geodata2/S2S/DL/GC_input/proportional/"
output_dir="/data/GC_output/proportional"

echo ========================================================
cat filelist.txt | while read input_file
do
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
    
    python ~/graphcast/GC_run.py --input "$input_file" --output "$output_file" --model "$model" --eval_steps "$eval_steps"
    chmod 777 "$output_file"
    echo "========================================================"
done 
