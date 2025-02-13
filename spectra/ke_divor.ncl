function get_gridtype(lat)
local d1,d2
begin
  gauss_grid = -1
  if (lat(0) .lt. -89.99999 ) then
    print ("detected equal angle grid with poles")
    gauss_grid = 2    ; use shaec routines
  else
    d1=lat(1)-lat(0)
    d2=lat(2)-lat(1)
    ;print(d1-d2)
    if (abs(d1-d2) .lt. 1e-8 ) then
      print ("detected equal angle grid without poles")
      gauss_grid = 3    ; need to interpolate using fo2fsh
    else
      print("detected Gauss grid")
      gauss_grid = 1  
    end if
  end if
  print("gauss_grid: "+gauss_grid)
  return(gauss_grid)
end



begin
if (isvar("input")) then 
  f = addfile(input, "r")
end if

gauss_grid=get_gridtype(f->lat)
; 1

div = f->sd
vor = f->svo

nlat = dimsizes(f->lat)
nlon = dimsizes(f->lon)

if (isvar("t")) then
  ntime = dimsizes(f->time)
else
  ntime = 1
end if

if (isvar("lev")) then
  nlevel = dimsizes(f->level)
else
  nlevel = 1
end if

print ("nlat x nlon "+nlat+"x"+nlon)

; Assume that lon is even
N = min((/nlat, (nlon+2)/2/))

if (.not. isvar("t")) then
  spectrum = new((/nlevel,N/), "double", 1d-99)
  spectrum!0 = "level"
  spectrum!1 = "N"

else if (.not. isvar("lev")) then
  spectrum = new((/ntime,N/), "double", 1d-99)
  spectrum!0 = "time"
  spectrum!1 = "N"

else
  spectrum = new((/ntime,nlevel,N/), "double", 1d-99)
  spectrum!0 = "time"
  spectrum!1 = "level"
  spectrum!2 = "N"
  end if
end if


do time=0,ntime-1
print("time: "+time)
  do level=0,nlevel-1
  print("level: "+level)

if (.not. isvar("t")) then
  var  = div(level, :, :) ;(time, lat, lon) -> (lat, lon)
  var2 = vor(level, :, :)

else if (.not. isvar("lev")) then
  var  = div(time, :, :) ;(time, lat, lon) -> (lat, lon)
  var2 = vor(time, :, :)

else
  print("("+time+","+level+",:,:)")
  var  = div(time, level, :, :) ;(time, lat, lon) -> (lat, lon)
  var2 = vor(time, level, :, :)
  end if
end if


spc = new ( (/nlat/), "double", 1d-99 )
spc = spc@_FillValue
spc!0 = "wavenumber"
x=spc
spc_vr = spc
spc_dv = spc

vr=var   ; vr는 소용돌이, var가 이미 VOR 자료임
dv=var2  ; dv는 발산, var2가 이미 DIV 자료임

; print("input dimension: "+dimsizes(vr))
ab_vr = shagC (vr(:,:))       ; 소용돌이 계수
ab_dv = shagC (dv(:,:))       ; 발산 계수

; print("ab_vr dimension: "+dimsizes(ab_vr))
tmp=dimsizes(ab_dv) ; (2,720,720)
nmax = tmp(1) ;720
mmax = tmp(2) ;720

; 소용돌이(vorticity) 성분: 실수부와 허수부의 제곱의 합
cr = ab_vr(0,:,:)                ; 실수 계수 (nlat x nlat 배열)
ci = ab_vr(1,:,:)                ; 허수 계수
pwr_vr = (cr^2 + ci^2)            ; 소용돌이 에너지
; 발산(divergence) 성분도 동일하게 계산
cr = ab_dv(0,:,:)                
ci = ab_dv(1,:,:)                
pwr_dv = (cr^2 + ci^2)            ; 발산 에너지

; 각 파수(n1)별로 m=0 모드와 m>0 모드들의 에너지를 합산
do n1=0,nlat-1
  x(n1) = n1                     ; x축: 구면 조화 차수 (혹은 파수)
  spc_vr(n1) = (/pwr_vr(n1,0) /) ; m=0인 경우 단독으로 사용
  spc_dv(n1) = (/pwr_dv(n1,0) /)
  factor = 4. * n1*(n1+1)        ; 스케일링 인자 (구면 조화 정규화 관련)
  ; m=1부터 최소( n1, mmax-1 )까지 에너지를 2배 가중치로 합산
  do m=1,min( (/n1,mmax-1/) )
    spc_vr(n1) = spc_vr(n1) + 2.*(pwr_vr(n1,m))
    spc_dv(n1) = spc_dv(n1) + 2.*(pwr_dv(n1,m))
  end do
  ; n1이 0이 아닐 경우, 지구의 반지름(6371229 m)을 이용하여 스케일링 적용
  if (n1 .ne. 0) then
    spc_vr(n1) = (6371229.^2) / factor * spc_vr(n1)  ; NCAR/TN-388+STR 기준 스케일
    spc_dv(n1) = (6371229.^2) / factor * spc_dv(n1)
  end if
end do
; 보정: Vector Spherical Harmonics 코드와 일치하도록 2로 나눔

; 수식으로 봤을 때 이상함. 제거해보기
; spc_vr = spc_vr/2
; spc_dv = spc_dv/2
; 전체 KE 스펙트럼: 소용돌이와 발산 에너지를 합산
spc = spc_vr + spc_dv 

print("Calculate spc")

;----------------------------
; spc와 x 배열을 NetCDF 파일에 저장
;----------------------------
; 수식으로 봤을 때 이상함. time
; spc_
; spc_dv = spc_dv/2
if (.not. isvar("t")) then
  spectrum(level,:) = spc(:)

else
  if (.not. isvar("lev")) then
  spectrum(time,:) = spc(:)

else
  spectrum(time,level,:) = spc(:)
  end if
end if

end do
end do


filename="output_divor.nc"
if (isvar("output")) then
  filename=output
end if 
system("/bin/rm "+filename)
out = addfile(filename, "c")   ; "c"는 새 파일 생성 (create) 모드
out->spectrum = spectrum
print("spc와 x를 "+filename+"에 저장하였습니다.")

end