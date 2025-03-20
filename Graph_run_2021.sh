#!/bin/bash

model="original"
eval_steps=60

input_dir="/geodata2/S2S/DL/GC_input/2021/"
dir="/geodata2/S2S/DL/GC_output/2021/"

start_date="2021-01-04"
num_mondays=52

echo "========================================================"
for (( i=0; i<num_mondays; i++ )); do
    # 매주 7일씩 더해서 월요일 날짜 계산
    current_date=$(date -d "$start_date +$((i * 7)) days" +%Y-%m-%d)
    
    input_file="${input_dir}${current_date}.nc"
    if [ ! -f "$input_file" ]; then
        echo "입력 파일이 존재하지 않습니다: $input_file"
        continue
    fi

    filename=$(basename "$input_file")
    output_file="${dir}${filename}"

    if [ -f "$output_file" ]; then
        echo "출력 파일이 이미 존재합니다: $output_file"
        continue
    fi

    echo "Processing $input_file -> $output_file"
    python ~/graphcast/GC_run.py --input "$input_file" --output "$output_file" --model "$model" --eval_steps "$eval_steps"
    chmod 777 "$output_file"
    echo "========================================================"
done
