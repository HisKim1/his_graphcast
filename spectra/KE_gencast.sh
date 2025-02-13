#! /bin/bash

for ens in {1..50}
do
    input="/data/GC_output/analysis/NWP/IFS-ENS/2021-06-21_${ens}.nc"
    output="/data/GC_output/analysis/NWP/IFS-ENS/spectra_2021-06-21_${ens}.nc"

    cdo -P 40 selname,sd,svo -sp2gp,linear -uv2dv,linear -remapscon2,n360 $input $output

    ncl /home/hiskim1/graphcast/spectra/capitalist.ncl 't=2' 'input="'$output'"' 'output="'$output'"'
done    
