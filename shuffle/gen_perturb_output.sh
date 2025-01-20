#!/bin/bash

while true; do
    file_count=$(ls /geodata2/S2S/DL/GC_input/shuffle/ERA5_0.{3,35,4,45,5}_*_*.nc 2>/dev/null | wc -l)
    
    if [ $file_count -eq 125 ]; then
        echo "Total 125 files are generated."
        echo "Wait for 5 minutes..."
        break
    else
        echo "# Current File: $file_count"
    fi
done


model="original"
eval_steps=40

input_files=$(ls /geodata2/S2S/DL/GC_input/shuffle/ERA5_0.5_*_*.nc 2>/dev/null  | tee filelist.txt)
input_dir="/geodata2/S2S/DL/GC_input/shuffle/"
output_dir="/geodata2/S2S/DL/GC_output/shuffle"

echo ========================================================
tac filelist.txt | while read input_file
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
    rm -f /geodata2/S2S/DL/GC_output/shuffle/done_flag
    python ~/graphcast/GC_run.py --input "$input_file" --output "$output_file" --model "$model" --eval_steps "$eval_steps"
    chmod 777 $output_file
    touch /geodata2/S2S/DL/GC_output/shuffle/done_flag
    echo "========================================================"
done 
