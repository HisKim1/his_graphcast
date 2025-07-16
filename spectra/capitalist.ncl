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

nlevel = dimsizes(f->level)

print ("nlat x nlon "+nlat+"x"+nlon)

; Assume that lon is even
N = min((/nlat, (nlon+2)/2/))

if (.not. isvar("t")) then
  spectrum = new((/nlevel,N/), "double", 1d-99)
  spectrum!0 = "level"
  spectrum!1 = "N"

else
  spectrum = new((/ntime,nlevel,N/), "double", 1d-99)
  spectrum!0 = "time"
  spectrum!1 = "level"
  spectrum!2 = "N"
end if

max_worker=16
nactive=0
ndone=0

do time=0,ntime-1
print("time: "+time)
  do level=0,nlevel-1
  print("level: "+level)

;---------------------------------------------
;---------------------------------------------
dq = inttochar(34)
arguments = str_get_sq()+ "t="+time+str_get_sq()+" "+str_get_sq()+"l="+level+str_get_sq()+" "+str_get_sq()+"input="+dq+input+dq+str_get_sq() + ">/dev/null"
cmd = "ncl /home/hiskim1/graphcast/spectra/worker.ncl " + arguments
print("Launching: " + cmd)
pid = subprocess(cmd)
nactive = nactive + 1

do while (nactive .ge. max_worker)
        finished_pid = subprocess_wait(0, True)
        if (finished_pid .gt. 0) then
          nactive = nactive - 1
          ndone   = ndone + 1
          print("Completed task, pid: " + finished_pid)
          break
        end if
      end do
end do
end do

  do while (nactive .gt. 0)
  finished_pid = subprocess_wait(0, True)
  if (finished_pid .gt. 0) then
    nactive = nactive - 1
    ndone   = ndone + 1
    print("Completed task, pid: " + finished_pid)
  end if
end do
;---------------------------------------------


;----------------------------
; spc와 x 배열을 NetCDF 파일에 저장
;----------------------------
; 수식으로 봤을 때 이상함. time
; spc_
; spc_dv = spc_dv/2


do time=0,ntime-1
print("time: "+time)
  do level=0,nlevel-1
  print("level: "+level)

    result = addfile(time+"_"+level+".nc", "r")
    spc=result->spc

if (.not. isvar("t")) then
  spectrum(level,:) = spc(:)
else
  spectrum(time,level,:) = spc(:)
end if

system("rm -f "+time+"_"+level+".nc")

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