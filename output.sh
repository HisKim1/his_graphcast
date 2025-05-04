#!/bin/bash

# ===========================================
#  Step 1: 날짜 범위 설정 및 반복 (2021년 매주 월요일)
# ===========================================
start_date="2021-01-04"   # 2021년 첫 월요일
end_date="2022-01-01"     # 종료 기준 (2022년 1월 1일 이전까지)

current_date="$start_date"

while [[ "$current_date" < "$end_date" ]]; do
    # 각 step 시작 시간 기록
    step_start=$(date +%s)

    filename=$(date -d "$current_date" +%Y-%m-%d)
    
    echo "--------------------------------------------------"
    echo "Generating data for $filename"
    echo "--------------------------------------------------"
    rm -f /geodata2/S2S/DL/GC_output/2021/${filename}.nc
    python /home/hiskim1/graphcast/GC_run.py \
        --model original \
        --eval_steps 60 \
        --input /geodata2/S2S/DL/GC_input/2021/${filename}.nc \
        --output /geodata2/S2S/DL/GC_output/2021/${filename}.nc

    chmod 777 /geodata2/S2S/DL/GC_output/2021/${filename}.nc

    # 각 step 종료 시간 기록 후 소요 시간 계산
    step_end=$(date +%s)
    step_elapsed=$((step_end - step_start))
    echo "Step for $filename completed in $step_elapsed seconds."
    
    # 다음 월요일로 이동 (+7일)
    current_date=$(date -I -d "$current_date + 7 days")
done

