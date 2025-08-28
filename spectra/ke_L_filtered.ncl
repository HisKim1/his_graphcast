function get_gridtype(lat)
local d1,d2
begin
  gauss_grid = -1
  if (lat(0) .lt. -89.99999 ) then
    print ("detected equal angle grid with poles")
    gauss_grid = 2
  else
    d1=lat(1)-lat(0)
    d2=lat(2)-lat(1)
    if (abs(d1-d2) .lt. 1e-8 ) then
      print ("detected equal angle grid without poles")
      gauss_grid = 3
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
    nlevel = dimsizes(f->plev)
  else
    nlevel = 1
  end if

  print ("nlat x nlon "+nlat+"x"+nlon)

  N = min((/nlat, (nlon+2)/2/))

  if (.not. isvar("t")) then
    spectrum = new((/nlevel,N/), "double", 1d-99)
    spectrum!0 = "level"
    spectrum!1 = "N"
  else
    if (.not. isvar("lev")) then
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

    if (.not. isvar("t")) then
      filtered_div = new((/nlevel,nlat,nlon/), "double", 1d-99)
      filtered_div!0 = "level"
      filtered_div!1 = "lat"
      filtered_div!2 = "lon"

      filtered_vor = new((/nlevel,nlat,nlon/), "double", 1d-99)
      filtered_vor!0 = "level"
      filtered_vor!1 = "lat"
      filtered_vor!2 = "lon"
  else
    if (.not. isvar("lev")) then
      filtered_div = new((/ntime,nlat,nlon/), "double", 1d-99)
      filtered_div!0 = "time"
      filtered_div!1 = "lat"
      filtered_div!2 = "lon"

      filtered_vor = new((/ntime,nlat,nlon/), "double", 1d-99)
      filtered_vor!0 = "time"
      filtered_vor!1 = "lat"
      filtered_vor!2 = "lon"
    else
      filtered_div = new((/ntime,nlevel,nlat,nlon/), "double", 1d-99)
      filtered_div!0 = "time"
      filtered_div!1 = "level"
      filtered_div!2 = "lat"
      filtered_div!2 = "lon"

      filtered_vor = new((/ntime,nlevel,nlat,nlon/), "double", 1d-99)
      filtered_vor!0 = "time"
      filtered_vor!1 = "level"
      filtered_vor!2 = "lat"
      filtered_vor!2 = "lon"
    end if
  end if

  do time=0,ntime-1
    print("time: "+time)
    do level=0,nlevel-1
      print("level: "+level)

      if (.not. isvar("t")) then
        var  = div(level, :, :)
        var2 = vor(level, :, :)
      else
        if (.not. isvar("lev")) then
          var  = div(time, :, :)
          var2 = vor(time, :, :)
        else
          print("("+time+","+level+",:,:)")
          var  = div(time, level, :, :)
          var2 = vor(time, level, :, :)
        end if
      end if

      spc = new ( (/nlat/), "double", 1d-99 )
      spc = spc@_FillValue
      spc!0 = "wavenumber"
      x=spc
      spc_vr = spc
      spc_dv = spc

      vr=var
      dv=var2

      ab_vr = shagC (vr(:,:))
      ab_dv = shagC (dv(:,:))

      ;========================================================================
        ;       ;--- [추가] 역변환을 위한 구면조화 계수 필터링:
        ;       ;     wavenumber 61~100만 남기고 나머지는 0으로 설정
      do n1 = 0, nlat-1
        ;if (n1 .lt. 61 .or. n1 .gt. 100) then
        if (n1 .lt. 89) then
          ab_vr(:, :, n1) = 0.0
          ab_dv(:, :, n1) = 0.0
        end if
      end do
  
      lon_out = f->lon
      lat_out = f->lat
  
      ;--- 구면조화 역변환:
      ;     vr2uvg()와 dv2uvg() 함수는 NCL에서 각각 회전성 및 발산성 계수로부터 u,v를 복원합니다.
      inverse_vr = shsgC(ab_vr, nlon)
      inverse_dv = shsgC(ab_dv, nlon)

      inverse_vr!0 = "lat"
      inverse_vr!1 = "lon"

      inverse_dv!0 = "lat"
      inverse_dv!1 = "lon"
  
    ;   ;--- KE 필드 계산: KE = 0.5*(u^2 + v^2)
    ;   ke_field = 0.5*(u_field^2 + v_field^2)


        if (.not. isvar("t")) then
            filtered_vor(level,:, : ) = inverse_vr(:, :)
            filtered_div(level,:, : ) = inverse_dv(:, :)
            
        else
            if (.not. isvar("lev")) then
            filtered_vor(time,:, : ) = inverse_vr(:, :)
            filtered_div(time,:, : ) = inverse_dv(:, :)
            else
            filtered_vor(time, level,:, : ) = inverse_vr(:, :)
            filtered_div(time, level,:, : ) = inverse_dv(:, :)
            end if
        end if

    end do
  end do

  filename="output_divor.nc"

  if (isvar("output")) then 
    filename=output
  end if

  ; system("/bin/rm "+filename)
  ; out = addfile(filename, "c")
  ; out->spectrum = spectrum
  ; print("spc와 x를 "+filename+"에 저장하였습니다.")


  filename_ke = "filtered_ke.nc"

  if (isvar("output2")) then
    filename_ke = output2
  end if

  system("/bin/rm " + filename_ke)
  out_ke = addfile(filename_ke, "c")
  out_ke->svo = filtered_vor
  out_ke->sd = filtered_div
  out_ke->lon = lon_out
  out_ke->lat = lat_out
  out_ke->time = f->time
  print("Kinetic energy field saved to " + filename_ke)
end

; cdo -dv2uv,linear -gp2sp,linear filtered_ke.nc filtered_uv.nc 하면 끝.