
#!/bin/bash

dir="/geodata2/S2S/DL/GC_output/2021/"

start_date="2021-01-04"
num_mondays=52

for (( i=0; i<num_mondays; i++ )); do
    # 매주 7일씩 더해서 월요일 날짜 계산
    current_date=$(date -d "$start_date +$((i * 7)) days" +%Y-%m-%d)
    
    input_file="${dir}${current_date}.nc"
    if [ ! -f "$input_file" ]; then
        echo "입력 파일이 존재하지 않습니다: $input_file"
        continue
    fi

    filename=$(basename "$input_file")
    output_file="${dir}spectra_${filename}"


    cdo -P 16 selname,sd,svo -sp2gp,linear -uv2dv,linear -remapscon2,n360 -selname,u,v -chname,u_component_of_wind,u,v_component_of_wind,v -sellevel,300 - $input_file $output_file

    ncl /home/hiskim1/graphcast/spectra/capitalist.ncl 't=2' 'input="'$output_file'"' 'output="'$output_file'"'

done
