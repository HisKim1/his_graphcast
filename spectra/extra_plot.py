import xarray as xr
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse

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
HRES = xr.open_dataset("/geodata2/Gencast/HRES_uv.zarr", engine="zarr")\
        .sel(level=300)\
            .mean("initial_time")\
                .rename({"u_component_of_wind":"u", "v_component_of_wind":"v"})\
                .isel(time=args.day * 4)
                

print(f"{prefix} Loading ENS and GenCast")
ENS = xr.open_dataset("/geodata2/S2S/DL/GC_output/IFS-ENS/uv300hPa.nc")\
    .isel(time=args.day * 2)
        
# d-1 00z부터 시작
GenCast = xr.open_dataset("/geodata2/Gencast/uv_300hPa.nc")\
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
    KE = 0.5 * (model_data['u']**2 + model_data['v']**2)

    # gradient 계산
    dKE = mpcalc.gradient(KE) 

    # 튜플 언팩
    dKE_dy, dKE_dx = dKE

    # Gradient magnitude 계산
    mag = np.sqrt(dKE_dx.data**2 + dKE_dy.data**2)

    # 좌표
    lon2d, lat2d = np.meshgrid(dKE_dx['lon'], dKE_dx['lat'])

    fig = plt.figure(figsize=(14, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines(resolution='110m', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='white', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)

    vmax = np.nanpercentile(mag, 95)
    c = ax.pcolormesh(
        lon2d, lat2d, mag,
        cmap='jet',
        shading='auto',
        vmin=0, vmax=vmax,  # colorbar scale 설정
        transform=ccrs.PlateCarree()
    )

    cb = plt.colorbar(c, ax=ax, orientation='vertical', pad=0.02, aspect=30)
    cb.set_label('|∇KE| [1/m]')

    ax.set_title(f"{model_name} Gradient Magnitude of Kinetic Energy (|∇KE|)", fontsize=14)
    plt.tight_layout()
    plt.show()

    out_png = f"/home/hiskim1/grad_{model_name}_ens{args.ens}_d{args.day}_12z.png"
    print(f"{prefix} Saving figure → {out_png}")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")

    print(f"{prefix} All done✅")