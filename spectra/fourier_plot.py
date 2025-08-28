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

# ---------------------------------------------------------------------------
# 1) Compute & save KE + FFT split
# ---------------------------------------------------------------------------
print(f"{prefix} Loading HRES")
HRES = xr.open_zarr("/geodata2/S2S/DL/GC_output/2021-12-20_HRES.zarr", chunks={"lat": 721, "lon": 1440}) \
        .isel(time=args.day * 4)

print(f"{prefix} Loading ENS and GenCast")
ENS = xr.open_dataset("/geodata2/S2S/DL/GC_output/2021-12-20_ENS.nc") \
        .isel(time=args.day * 2)
GenCast = xr.open_zarr("/geodata2/S2S/DL/GC_output/2021-12-20_GEN.zarr", chunks={"lat": 721, "lon": 1440}) \
        .sel(level=300) \
        .rename({"u_component_of_wind":"u","v_component_of_wind":"v"}) \
        .isel(time=args.day * 2 - 1)

if args.ens == -1:
    args.ens = "mean"
    print(f"{prefix} Averaging over all members")
    ENS = ENS.mean("ensemble")
    GenCast = GenCast.mean("sample")
else:
    ENS = ENS.isel(ensemble=args.ens)
    GenCast = GenCast.isel(sample=args.ens - 1)

# import os
out_png = f"/home/hiskim1/2021-12-20_ens{args.ens}_d{args.day}_12z.png"
# if os.path.exists(out_png):
#     print(f"{prefix} File already exists → {out_png}")
#     exit()

print(f"{prefix} Computing kinetic energy")
HRES = kinetic_energy(HRES)
ENS  = kinetic_energy(ENS)
GenCast = kinetic_energy(GenCast)

k_thr = 100
print(f"{prefix} Performing FFT split at k={k_thr}")
HRES_high = fft_split_lon(HRES.compute(), k_thr, "high")
HRES_low  = fft_split_lon(HRES.compute(), k_thr, "low")
gen_high  = fft_split_lon(GenCast.compute(), k_thr, "high")
gen_low   = fft_split_lon(GenCast.compute(), k_thr, "low")
ens_high  = fft_split_lon(ENS.compute(), k_thr, "high")
ens_low   = fft_split_lon(ENS.compute(), k_thr, "low")

print(f"{prefix} Renaming outputs and building Dataset")
HRES_high.name = "ke_hres_high"
HRES_low.name  = "ke_hres_low"
gen_high.name  = "ke_genc_high"
gen_low.name   = "ke_genc_low"
ens_high.name  = "ke_ens_high"
ens_low.name   = "ke_ens_low"

ds = xr.Dataset({
    "ke_hres":      HRES,
    "ke_ens":       ENS,
    "ke_gen":       GenCast,
    "ke_hres_low":  HRES_low,
    "ke_hres_high": HRES_high,
    "ke_genc_low":  gen_low,
    "ke_genc_high": gen_high,
    "ke_ens_low":   ens_low,
    "ke_ens_high":  ens_high
}, coords={"lat": HRES.lat, "lon": HRES.lon})

out_nc = f"/geodata2/Gencast/fourier/211220_ens{args.ens}_d{args.day}_12z.nc"
print(f"{prefix} Saving netCDF → {out_nc}")
ds.to_netcdf(out_nc)
ds.close()
# ---------------------------------------------------------------------------
# 2) Plot
# ---------------------------------------------------------------------------
print(f"{prefix} Starting plotting")

from matplotlib.colors import SymLogNorm

ds = xr.open_dataset(out_nc)
# low_norm  = Normalize(vmin=0,   vmax=650)
low_norm  = Normalize(vmin=0,   vmax=3000)
# high_norm = PowerNorm(gamma=0.5, vmin=0, vmax=110)
high_norm = SymLogNorm(
    linthresh=15,
    linscale=1.0,
    vmin=-200,
    vmax= 200,
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

def mesh_plot(ax, data, norm, title, cmap="YlGnBu_r"):
    arr = data.values.copy()
    arr += 1e-6 * np.random.rand(*arr.shape)
    im = ax.contourf(
        data.lon, data.lat, arr,
        levels=40,              # 60단계로 고정
        cmap=cmap,
        norm=norm,              # 모든 패널에 동일한 norm 적용
        extend="both",
        transform=ccrs.PlateCarree()
    )
    ax.set_title(title, fontsize=16)
    ax.coastlines(linewidth=0.4)
    ax.set_xticks([]); ax.set_yticks([])
    return im

print(f"{prefix} Plotting panels")
mesh_plot(ax_ens_low,  ds.ke_ens_low,  low_norm,  f"IFS-ENS k≤{k_thr}")
mesh_plot(ax_ens_high, ds.ke_ens_high, high_norm, f"IFS-ENS k>{k_thr}", cmap="RdBu_r")
mesh_plot(ax_genc_low, ds.ke_genc_low, low_norm,  f"GenCast k≤{k_thr}")
mesh_plot(ax_genc_high,ds.ke_genc_high, high_norm, f"GenCast k>{k_thr}", cmap="RdBu_r")
mesh_plot(ax_hres_low, ds.ke_hres_low, low_norm, f"IFS-HRES k≤{k_thr}")
mesh_plot(ax_hres_high,ds.ke_hres_high,high_norm, f"IFS-HRES k>{k_thr}", cmap="RdBu_r")

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
