#!/bin/bash

# 파일 경로 패턴 저장

while true; do
    # 현재 파일 개수 확인
    file_count=$(ls /geodata2/S2S/DL/GC_input/shuffle/ERA5_0.{01,05,1,15,2,25}_*_*.nc 2>/dev/null | wc -l)
    
    if [ $file_count -eq 150 ]; then
        echo "Total 150 files are generated."
        break
    else
        echo "# Current File: $file_count"
    fi
done


model="original"
eval_steps=40

input_files=$(ls /geodata2/S2S/DL/GC_input/proportional/ERA5_0.5_*_*.nc | tee filelist.txt)
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

rm -f filelist.txt
