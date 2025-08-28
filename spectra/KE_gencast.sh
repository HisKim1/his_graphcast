#! /bin/bash

for ens in {0..9}
do
    input="/geodata2/Gencast/uv_300hPa/${ens}.nc"
    output="/geodata2/Gencast/uv_300hPa/spectra_${ens}.nc"
    output2="/geodata2/Gencast/uv_300hPa/H_${ens}.nc"
    output3="/geodata2/Gencast/uv_300hPa/L_${ens}.nc"

    cdo -P 40 selname,sd,svo -sp2gp,linear -uv2dv,linear -remapscon2,n360 -chname,u_component_of_wind,u,v_component_of_wind,v $input $output

    ncl /home/hiskim1/graphcast/spectra/ke_H_filtered.ncl 't=2' 'input="'$output'"' 'output2="'$output2'"' & 

    ncl /home/hiskim1/graphcast/spectra/ke_L_filtered.ncl 't=2' 'input="'$output'"' 'output2="'$output3'"' &
done    