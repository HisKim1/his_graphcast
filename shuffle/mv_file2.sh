#!/bin/bash

SOURCE_DIR="/geodata2/S2S/DL/GC_output/shuffle"
TARGET_DIR="/data/GC_output/shuffle"

echo "=== 파일 이동 스크립트 시작: $(date) ==="

# 무한 반복
while true
do
    echo "------"
    echo "[INFO] 새로운 파일(권한 777) 확인 중: $(date)"
    echo "------"
    
    # GC_0.{3,35,4,45,5}_*.nc 패턴에 해당하는 모든 파일을 순회
    for file in ${SOURCE_DIR}/GC_0.{5,45,4,35,3}_*.nc
    do
        # 파일이 실제로 존재하는지 확인 (패턴에 맞는 파일이 없으면 $file = 문자열 그대로가 될 수 있음)
        if [ ! -e "$file" ]; then
            # 매칭되는 파일이 없는 경우 별다른 메시지 없이 건너뛴다.
            continue
        fi

        # 파일의 권한을 가져와서(숫자 모드) 777인지 확인
        perm=$(stat -c '%a' "$file" 2>/dev/null)
        if [ $? -ne 0 ]; then
            echo "[ERROR] 권한 확인 실패: $file" >&2
            continue
        fi
        
        echo "[INFO] 확인 중 파일: $file, 권한: $perm"
        if [ "$perm" = "777" ]; then
            echo "[INFO] 권한이 777이므로 이동을 시도합니다."
            if mv -v "$file" "${TARGET_DIR}"; then
                echo "[INFO] 이동 성공: $file -> $TARGET_DIR"
            else
                echo "[ERROR] 이동 실패: $file -> $TARGET_DIR" >&2
            fi
        else
            echo "[INFO] 권한이 $perm 이므로 이동하지 않음: $file"
        fi
    done

    # 다음 확인 전까지 대기(예: 60초)
    echo "[INFO] 60초 대기 후 다시 확인합니다."
    sleep 60
done
