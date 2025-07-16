#! /bin/bash

for ens in {1..50}
do
    input="/data/GC_output/analysis/NWP/IFS-ENS/2021-06-21/${ens}.nc"
    output="/data/GC_output/analysis/NWP/IFS-ENS/2021-06-21/spectra_${ens}.nc"

    cdo -P 40 selname,sd,svo -sp2gp,linear -uv2dv,linear -remapscon2,n360 -chlevel,30000,300 -chname,plev,level $input $output

    ncl /home/hiskim1/graphcast/spectra/capitalist.ncl 't=2' 'input="'$output'"' 'output="'$output'"'
done    
