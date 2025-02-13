#!/bin/bash

# Usage: ./KE_input.sh input.nc output.nc
# e.g. bash KE_input.sh /geodata2/Gencast/era5_2021-06-21.nc /geodata2/Gencast/0210output.nc

ncl_input=$1
ncl_output=$2

rm $2

# remapbil
# perform a bilinear interpolation on all input fields
# n360: 
# cdo -P 40 remapbil,n360 $ncl_input remap_n360.nc

# uv2dv
# U and V wind to divergence and vorticity
# output: spherical harmonic coefficients of div and vor
# linear: the shortest wavelength is represented by 2 grid points
# Requirement
# need to have the names "u" and "v" or code numbers "131" and "132"
# cdo uv2dv,linear remap_n360.nc uv2dv_linear.nc

# sp2gp
# spectral coefficients to regular Gaussian grid
# linear: same with above
# cdo -P 40 sp2gp,linear uv2dv_linear.nc sp2gp_linear.nc

# cdo selname,sd,svo sp2gp_linear.nc $ncl_output

# rm -f remap_n360.nc uv2dv_linear.nc sp2gp_linear.nc 

# cdo -P 40 selname,sd,svo -sp2gp,linear -uv2dv,linear -remapbil,n180 -chname,u_component_of_wind,u,v_component_of_wind,v -delete,name=batch $ncl_input $ncl_output

cdo -P 40 selname,sd,svo -sp2gp,linear -uv2dv,linear -remapbil,n360 -chname,u_component_of_wind,u,v_component_of_wind,v $ncl_input $ncl_output

ncl /home/hiskim1/graphcast/spectra/ke_divor.ncl 't=2' 'input="'$ncl_output'"' 'output="'$ncl_output'"'