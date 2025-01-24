import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.switch_backend('Agg')

VARIABLE_UNIT = {
    "geopotential_at_surface": "m**2 s**-2",
    "land_sea_mask": "(0 - 1)",
    "toa_incident_solar_radiation": "J m**-2",
    "10m_u_component_of_wind": "m s**-1",
    "10m_v_component_of_wind": "m s**-1",
    "2m_temperature": "K",
    "mean_sea_level_pressure": "hPa",
    "total_precipitation_6hr": "mm",
    "total_precipitation_12hr": "mm",
    "geopotential": "m**2 s**-2",
    "specific_humidity": "kg kg**-1",
    "temperature": "K",
    "u_component_of_wind": "m s**-1",
    "v_component_of_wind": "m s**-1",
    "vertical_velocity": "Pa s**-1",
    'toa_incident_solar_radiation': 'J m**-2',
}

PLOT_TYPE = {
    "platecarree": ccrs.PlateCarree(central_longitude=180),
    "mercator": ccrs.Mercator(central_longitude=180),
    "mollweide": ccrs.Mollweide(central_longitude=180),
    "robinson": ccrs.Robinson(central_longitude=180),
    "orthographic": ccrs.Orthographic(central_longitude=0, central_latitude=0),
    "stereographic": ccrs.Stereographic(central_longitude=0, central_latitude=90),
    "lambertconformal": ccrs.LambertConformal(central_longitude=0, central_latitude=0),
    "northpolarstereo": ccrs.NorthPolarStereo(),
    "southpolarstereo": ccrs.SouthPolarStereo(),
    "geostationary": ccrs.Geostationary(central_longitude=0),
    "goodehomolosine": ccrs.InterruptedGoodeHomolosine(central_longitude=0),
    "eurasiacentric": ccrs.EuroPP(),
    "millercylindrical": ccrs.Miller(central_longitude=180),
    "equidistantconic": ccrs.EquidistantConic(central_longitude=0, central_latitude=0),
    "azimuthalequidistant": ccrs.AzimuthalEquidistant(central_longitude=0, central_latitude=0)
}


def plot(args):
    """
        plot dataset via multiprocessing

        Args:
            dataset: xarray dataset which regions, time, level, and variable is specified for memory efficiency
            target_var: target variable to plot
            plot_type: type of plot; key value in PLOT_TYPE/ e.g. 'platecarree'
            color: color map / e.g. 'RdBu_r'
            title: title of the plot
            file_path: file name to save the plot
            norm: normalization for color map / e.g. TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

        Usage:
            with Pool() as pool:
                pool.map(plot, arg_list)
    """
    dataset, target_var, plot_type, color, title, file_path, norm = args
    
    if "latitude" in dataset.dims:
        dataset = dataset.rename({"latitude": "lat", "longitude": "lon"})

    fig = plt.figure(figsize=(20,10))
    ax = plt.axes(projection=PLOT_TYPE[plot_type])
    
    if norm is None:
        print("WARNING: No normalization is provided. Using default normalization.", 
              "If a partial dataset is provided, the normalization may not be accurate.")
        
        weights = np.cos(np.deg2rad(dataset.lat))
        weights.name = "weights"
        weighted = dataset.weighted(weights)
        mean_val = weighted.mean(('lat', 'lon'))
        # max_val = np.abs(dataset).max().values
        std_val = weighted.std(('lat', 'lon')).max()
        print(f"mean: {mean_val.values}, std: {std_val.values}")
        norm = TwoSlopeNorm(vmin=mean_val.values - std_val.values, vcenter=mean_val.values, vmax=mean_val.values + std_val.values) #     <--------------- FIX vcenter value

    # 차이 데이터를 지도에 그립니다.
    im = ax.pcolormesh(dataset.lon, dataset.lat, dataset.squeeze().values, 
                    transform=ccrs.PlateCarree(), 
                    cmap=color,
                    norm=norm,
                    shading='auto')

    cbar = plt.colorbar(im, ax=ax, extend='both',
                        label=f'{target_var} {VARIABLE_UNIT[target_var]}')
        
    # 지도 특성 추가
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.gridlines(draw_labels=True)

    
    plt.title(f'{title}')   
    
    plt.savefig(f'{file_path}', dpi=300, bbox_inches='tight')
    plt.close("all")
    # print(f'{file_path} saved')


def save_gif(image_frames, save_path, duration):
    image_frames[0].save(save_path,
                         format='GIF',
                         append_images=image_frames[1:],
                         # save_all : 모든 프레임을 저장할 것인지
                         save_all=True,
                         # duration : 프레임 간의 시간 간격 (ms)
                         duration=duration,
                         loop=0)