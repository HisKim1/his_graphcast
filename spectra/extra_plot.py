import xarray as xr
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse
import matplotlib.gridspec as gridspec
import matplotlib.path as mpath
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--day", type=int, default=1)
parser.add_argument("--ens", type=int, default=1)
args = parser.parse_args()

prefix = f"day {args.day} | ens {args.ens}:"
print(f"{prefix} Starting script")


# ---------------------------------------------------------------------------
# 1) Compute & save KE + FFT split
# ---------------------------------------------------------------------------
print(f"{prefix} Loading HRES")
HRES = xr.open_zarr("/geodata2/Gencast/HRES_uv.zarr", decode_timedelta=True)\
        .sel(level=300)\
            .mean("initial_time")\
                .rename({"u_component_of_wind":"u", "v_component_of_wind":"v"})\
                .isel(time=args.day * 4)
                

print(f"{prefix} Loading ENS and GenCast")
ENS = xr.open_dataset("/geodata2/S2S/DL/GC_output/IFS-ENS/uv300hPa.nc", decode_timedelta=True)\
    .isel(time=args.day * 2)
        
# d-1 00z부터 시작
GenCast = xr.open_dataset("/geodata2/Gencast/uv_300hPa.nc", decode_timedelta=True)\
    .rename({"u_component_of_wind":"u", "v_component_of_wind":"v"})\
    .isel(time=args.day * 2 - 1)

if args.ens == -1:
    GenCast = GenCast.mean("sample")
    ENS = ENS.mean("ensemble")
    args.ens = "mean"

else:
    ENS = ENS.isel(ensemble = args.ens)
    GenCast = GenCast.isel(sample=args.ens - 1)

model = {
    "HRES": HRES,
    "ENS": ENS,
    "GenCast": GenCast
}

for model_name, model_data in model.items():
    # NaN이면 0으로 처리
    model_data = model_data.fillna(0)

    KE = 0.5 * (model_data['u']**2 + model_data['v']**2)
    lat = KE.lat.values
    lon = KE.lon.values
    # gradient 계산
    dKE = mpcalc.gradient(KE) 

    # 튜플 언팩
    dKE_dy, dKE_dx = mpcalc.geospatial_gradient(KE)


    dKE_dy = np.asarray(dKE_dy)   # memoryview → ndarray
    dKE_dx = np.asarray(dKE_dx)

    mag = np.sqrt(dKE_dx**2 + dKE_dy**2)

    magnitude = xr.DataArray(mag, coords=KE.coords, dims=['lat', 'lon'])

    # 좌표

    band_start = 130.5  # 혹은 직접 지정 ex: np.where((lon>=120) & (lon<150))[0]
    band_slice = slice(band_start, band_start + 9)

    band_vals = magnitude.sel(lon=band_slice).mean("lon")

    # Figure 레이아웃
    fig = plt.figure(figsize=(16,6), constrained_layout=True)
    gs = gridspec.GridSpec(1, 21, figure=fig, width_ratios=[1,1,1,1,1, 0.3,
                                                            2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                                                            0.6])
    ax_prof = fig.add_subplot(gs[0, 0:5])
    ax_grid = fig.add_subplot(gs[0, 6:20], projection=ccrs.PlateCarree())  # 마지막(20) 제외
    cax     = fig.add_subplot(gs[0, 20])                                   # colorbar 전용 축

    ax_prof.plot(band_vals, KE.lat, lw=1, color='b')
    ax_prof.set_ylabel('Latitude')
    ax_prof.set_xlabel(r'$|\nabla KE|$ [m s$^{-2}$]')
    ax_prof.invert_xaxis()
    # x_max = np.nanpercentile(band_vals, 99)
    ax_prof.set_xlim(0.0012, 0)
    ax_prof.ticklabel_format(axis='x', style='sci', scilimits=(-4,2))
    ax_prof.set_ylim(lat.min(), lat.max())
    ax_prof.grid(alpha=0.3)
    # (선택) 위도 0, ±30, ±60 선만 ytick
    ticks = [-60, -30, 0, 30, 60]
    ax_prof.set_yticks(ticks)
    # (2) 오른쪽: 지도
    if hasattr(mag, "magnitude"):
        mag_plot = mag.magnitude
    else:
        mag_plot = mag
    vmax = np.nanpercentile(mag_plot, 95)
    pcm = ax_grid.pcolormesh(lon, lat, mag_plot, 
                            cmap='jet', shading='auto', vmin=0, vmax=vmax,
                            transform=ccrs.PlateCarree()
                            )
    ax_grid.coastlines(resolution='110m', linewidth=0.8, color='black')
    ax_grid.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax_grid.add_feature(cfeature.LAND, facecolor='white', zorder=0)
    ax_grid.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)

    ax_grid.set_xlabel('Longitude')
    ax_grid.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

    ax_grid.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax_grid.set_yticks(np.arange(-80,   81, 20), crs=ccrs.PlateCarree())
    gl = ax_grid.gridlines(draw_labels=False, linestyle='--', linewidth=0.5, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size':9}
    gl.ylabel_style = {'size':9}

    # axis labels (optional; gridline labels generally suffice)
    ax_grid.set_xlabel('Longitude')
    ax_grid.set_ylabel('')
    ax_grid.yaxis.set_tick_params(labelleft=False) 

    cb = fig.colorbar(pcm, cax=cax, orientation='vertical')
    cb.set_label(r'$|\nabla KE|$ [m s$^{-2}$]')
    plt.savefig(f"/home/hiskim1/extra_plot_{model_name}_{args.day}_{args.ens}.png")