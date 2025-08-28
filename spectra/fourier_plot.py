#!/usr/bin/env python3

import argparse
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
import cartopy.crs as ccrs

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
# Helper functions
# ---------------------------------------------------------------------------
def kinetic_energy(ds):
    return 0.5 * (ds["u"] ** 2 + ds["v"] ** 2)

def fft_split_lon(da, k_thr, part="low"):
    lon_dim = da.dims[-1]
    n_lon   = da.sizes[lon_dim]
    k_idx   = np.fft.fftfreq(n_lon) * n_lon
    keep    = np.abs(k_idx) <= k_thr if part == "low" else np.abs(k_idx) > k_thr

    fft_arr = np.fft.fft(da, axis=-1)
    fft_arr[..., ~keep] = 0
    filtered = np.fft.ifft(fft_arr, axis=-1).real

    return xr.DataArray(
        filtered, coords=da.coords, dims=da.dims,
        name=f"{da.name}_k{'le' if part=='low' else 'gt'}{k_thr}"
    )


out_nc = f"/geodata2/Gencast/fourier/fourier_ens{args.ens}_d{args.day}_12z.nc"
import os
if not os.path.exists(out_nc):
    # ---------------------------------------------------------------------------
    # 1) Compute & save KE + FFT split
    # ---------------------------------------------------------------------------
    print(f"{prefix} Loading HRES")
    HRES = xr.open_zarr("/geodata2/Gencast/HRES_uv.zarr")\
            .sel(level=300)\
                .mean("initial_time")\
                    .rename({"u_component_of_wind":"u", "v_component_of_wind":"v"})\
                    .isel(time=args.day * 4)
                    

    print(f"{prefix} Loading ENS and GenCast")
    ENS = xr.open_dataset("/geodata2/S2S/DL/GC_output/IFS-ENS/uv300hPa.nc", chunks={"lat":1440, "lon":721})\
        .isel(time=args.day * 2)
            
    # d-1 00z부터 시작
    GenCast = xr.open_dataset("/geodata2/Gencast/uv_300hPa.nc", chunks={"lat":1440, "lon":721})\
        .rename({"u_component_of_wind":"u", "v_component_of_wind":"v"})\
        .isel(time=args.day * 2 - 1)

    if args.ens == -1:
        GenCast = GenCast.mean("sample")
        ENS = ENS.mean("ensemble")
        args.ens = "mean"

    else:
        ENS = ENS.isel(ensemble = args.ens)
        GenCast = GenCast.isel(sample=args.ens - 1)


    # import os
    out_png = f"/home/hiskim1/fourier_ens{args.ens}_d{args.day}_12z.png"

    print(f"{prefix} Computing kinetic energy")
    HRES = kinetic_energy(HRES)
    ENS = kinetic_energy(ENS)
    GenCast = kinetic_energy(GenCast)

    k_thr = 100
    print(f"{prefix} Performing FFT split at k={k_thr}")
    HRES_high = fft_split_lon(HRES.compute(), k_thr, "high")
    HRES_low = fft_split_lon(HRES.compute(), k_thr, "low")

    gen_high = fft_split_lon(GenCast.compute(), k_thr, "high")
    gen_low = fft_split_lon(GenCast.compute(), k_thr, "low")

    ens_high = fft_split_lon(ENS.compute(), k_thr, "high")
    ens_low = fft_split_lon(ENS.compute(), k_thr, "low")

    print(f"{prefix} Renaming outputs and building Dataset")
    HRES_high.name = "ke_hres_high"
    HRES_low.name = "ke_hres_low"

    gen_high.name = "ke_gen_high"
    gen_low.name = "ke_gen_low"

    ens_high.name = "ke_ens_high"
    ens_low.name = "ke_ens_low"

    ds = xr.Dataset({
        "ke_hres": HRES,
        "ke_ens": ENS,
        "ke_gen": GenCast,
        "ke_hres_low": HRES_low,
        "ke_hres_high": HRES_high,
        "ke_genc_low": gen_low,
        "ke_genc_high": gen_high,
        "ke_ens_low": ens_low,
        "ke_ens_high": ens_high
    }, coords={
        "lat": HRES.lat,
        "lon": HRES.lon
    })
    ds.to_netcdf(out_nc)
    ds.close()

# ---------------------------------------------------------------------------
# 2) Plot
# ---------------------------------------------------------------------------
print(f"{prefix} Starting plotting")

from matplotlib.colors import SymLogNorm

ds = xr.open_dataset(out_nc)
# low_norm  = Normalize(vmin=0,   vmax=650)
low_norm  = Normalize(vmin=0,   vmax=650)
# high_norm = PowerNorm(gamma=0.5, vmin=0, vmax=110)
high_norm = SymLogNorm(
    linthresh=10,
    linscale=1.0,
    vmin=-70,
    vmax= 70,
    base=10
)

fig, axes = plt.subplots(
    3, 2,
    figsize=(12, 9),
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'wspace': 0.2, 'hspace': 0.2}
)
(ax_hres_low, ax_hres_high,
 ax_ens_low,  ax_ens_high,
 ax_genc_low, ax_genc_high) = axes.flatten()

import matplotlib.ticker as mticker

def mesh_plot(ax, data, norm, cmap="YlGnBu_r"):
    arr = data.values.copy()
    arr += 1e-6 * np.random.rand(*arr.shape)
    im = ax.contourf(
        data.lon, data.lat, arr,
        levels=40,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=ccrs.PlateCarree()
    )
    ax.coastlines(linewidth=0.4)

    # 공통 tick 위치(숫자만), gridline(라벨 없음)
    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-80,   81, 20), crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=False, linestyle='--', linewidth=0.4, alpha=0.5)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    return im


print(f"{prefix} Plotting panels")
mesh_plot(ax_ens_low,   ds.ke_ens_low,   low_norm)
mesh_plot(ax_ens_high,  ds.ke_ens_high,  high_norm, cmap="RdBu_r")
mesh_plot(ax_genc_low,  ds.ke_genc_low,  low_norm)
mesh_plot(ax_genc_high, ds.ke_genc_high, high_norm, cmap="RdBu_r")
mesh_plot(ax_hres_low,  ds.ke_hres_low,  low_norm)
mesh_plot(ax_hres_high, ds.ke_hres_high, high_norm, cmap="RdBu_r")

# 패널 레이블
panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
for lab, ax in zip(panel_labels, [ax_hres_low, ax_hres_high,
                                  ax_ens_low,  ax_ens_high,
                                  ax_genc_low, ax_genc_high]):
    ax.text(0.015, 0.965, lab, transform=ax.transAxes,
            ha='left', va='top', fontsize=13, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=2.0))

# axes를 2D로 다루기 편하게 재구성
axs = np.array([[ax_hres_low,  ax_hres_high],
                [ax_ens_low,   ax_ens_high],
                [ax_genc_low,  ax_genc_high]])

for i in range(3):
    for j in range(2):
        ax = axs[i, j]
        # 기본: tick은 유지
        ax.tick_params(axis='both', which='both', length=4, labelsize=9)

        if j == 0:
            ax.set_ylabel('Latitude')
        else:
            ax.set_ylabel('')
            ax.yaxis.set_tick_params(labelleft=False)  # 숫자 숨김 (틱 마크는 유지)

        if i == 2:
            ax.set_xlabel('Longitude')
        else:
            ax.set_xlabel('')
            ax.xaxis.set_tick_params(labelbottom=False)  # 숫자 숨김 (틱 마크는 유지)

print(f"{prefix} Adding colorbars")
cax_low  = fig.add_axes([0.49, 0.25, 0.01, 0.5])
fig.colorbar(plt.cm.ScalarMappable(norm=low_norm, cmap="YlGnBu_r"), cax=cax_low, extend="max")
cax_high = fig.add_axes([0.91, 0.25, 0.01, 0.5])
fig.colorbar(plt.cm.ScalarMappable(norm=high_norm, cmap="RdBu_r"),
             cax=cax_high, label="KE (m$^2$ s$^{-2}$)",
             extend="both")

print(f"{prefix} Saving figure → {out_png}")
plt.savefig(out_png, dpi=300, bbox_inches="tight")

print(f"{prefix} All done✅")
