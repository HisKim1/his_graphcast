#!/bin/bash

# 파일명 설정
input_file="dataset.nc"
gaussian_file="dataset_gaussian.nc"
div_spectral_file="div_spectral.nc"
vo_spectral_file="vo_spectral.nc"
dke_spectral_file="dke_spectral.nc"

# Step 1: Gaussian Grid Interpolation (N360)
# remap을 사용하여 Gaussian 그리드로 보간 (N360 해상도)
cdo remap,n360 $input_file $gaussian_file

# Step 2: Calculate Divergence and Vorticity
# div와 vor을 각각 사용하여 발산과 소용돌이 계산
cdo div $gaussian_file $div_spectral_file
cdo vor $gaussian_file $vo_spectral_file

# Step 3: Calculate DKE (Difference Kinetic Energy) Spectrum
# 발산과 소용돌이의 제곱을 더하고 제곱근을 취하여 DKE 계산
cdo sqrt -add -sqr $div_spectral_file -sqr $vo_spectral_file $dke_spectral_file
