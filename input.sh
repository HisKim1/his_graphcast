#!/bin/bash

# ===========================================
#  Step 0: 초기화 (tmp 디렉토리 비우기)
# ===========================================
echo "Clearing /geodata2/S2S/DL/GC_input/tmp directory..."
# rm -rf /geodata2/S2S/DL/GC_input/tmp/*

# ===========================================
#  Step 1: 날짜 범위 설정 및 반복 (2021년 매주 월요일)
# ===========================================
start_date="2021-01-04"   # 2021년 첫 월요일
end_date="2022-01-01"     # 종료 기준 (2022년 1월 1일 이전까지)

current_date="$start_date"

while [[ "$current_date" < "$end_date" ]]; do
    # month와 day는 앞의 0을 제거한 정수형으로 추출
    month=$(date -d "$current_date" +%-m)
    day=$(date -d "$current_date" +%-d)
    filename=$(date -d "$current_date" +%Y-%m-%d)
    
    echo "--------------------------------------------------"
    echo "Generating data for $filename"
    echo "--------------------------------------------------"
    
    python /home/hiskim1/graphcast/GraphCast_input.py \
        --year 2021 \
        --month $month \
        --day $day \
        --output /geodata2/S2S/DL/GC_input/2021/${filename}.nc

    chmod 777 /geodata2/S2S/DL/GC_input/2021/${filename}.nc

    # 처리 후 tmp 디렉토리 초기화
    rm -rf /geodata2/S2S/DL/GC_input/tmp/*

    # 다음 월요일로 이동 (+7일)
    current_date=$(date -I -d "$current_date + 7 days")
done

echo "All data generation completed!"
