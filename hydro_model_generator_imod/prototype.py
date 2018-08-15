from functools import partial
from collections import OrderedDict
from rasterio import features
import os
import xarray as xr
import imod
import numpy as np
import shapely.geometry as sg
import pyproj
import shapely
import requests
import json
from hydro_model_builder import model_builder
from hydro_model_builder import cli
from hydro_model_generator_wflow import hydro_model_generator_wflow as hmw


def utm_epsg(region):
    """Return UTM EPSG code for a given region geojson feature"""
    centroid = sg.shape(region).centroid
    longitude, latitude = centroid.x, centroid.y
    
    # northern latitudes
    if latitude > 0:
        UTMzone = int((np.floor((longitude + 180.0) / 6.0) % 60.0) + 1)
        epsg = 32600 + UTMzone
    # southern latitudes
    else:
        UTMzone = int((np.floor((longitude + 180.0) / 6.0) % 60.0) + 1)
        epsg = 32700 + UTMzone
        
    return epsg


def general_options(region, d):
    # get data from hydro-engine one by one
    defaults = d["hydro-engine"]["defaults"]
    for ds_override in d["hydro-engine"]["datasets"]:
        ds = defaults.copy()
        ds.update(ds_override)
        if ds["source"] == "earth-engine":
            get_hydro_data(region, ds)
        else:
            # TODO support non earth-engine datasets
            print("skipped variable:", ds["variable"])

            
def ne_features(path):
    with open(path) as f:
        js = json.load(f)
    features = [sg.shape(f["geometry"]) for f in js["features"]]
    return sg.MultiPolygon(features)


def ne_rivers(path, buffer=0.000000001):
    """ 
    Lame function to do something with rivers
    """
    with open(path) as f:
        js = json.load(f)
        
    features = [sg.shape(f["geometry"]) for f in js["features"]]
    return sg.MultiPolygon(features)
    
    rivers = []
    for f in js["features"]:
        linestring = sg.shape(f["geometry"])
        polygon = linestring.buffer(buffer)
        rivers.append(polygon)

    # add an extra buffer of 0, to create a valid geometry
    return sg.MultiPolygon(rivers).buffer(0.0)            


def reproject(shapes, src_proj, dst_proj):
    project = partial(
        pyproj.transform,
        src_proj,
        dst_proj,   
    )
    return shapely.ops.transform(project, shapes)


def rasterize(shapes, like, fill, **kwargs):
    """Rasterize a list of (geometry, fill_value) tuples onto the given
    xarray coordinates.
    """
       
    raster = features.rasterize(shapes,
                                out_shape=like.shape,
                                fill=fill,
                                transform = imod.util.transform(like),
                                **kwargs,
                                   )
    
    return xr.DataArray(raster, like.coords, like.dims)


def get_shapemask(path, region, proj):
    f = ne_features(path).intersection(sg.shape(region))
    src_proj = pyproj.Proj(init="EPSG:4326")
    dst_proj = proj
    return reproject(f, src_proj, dst_proj)    


def get_mask(path, region, like, fill, proj):
    shapes = get_shapemask(path, region, proj)
    # rasterize accepts only iterables
    try:
        iter(shapes)
    except TypeError:
        shapes = (shapes,)    
    return rasterize(shapes, like, fill)


def load_raster(path, like, region, epsg):
    xmin, ymin, xmax, ymax = sg.shape(region).bounds
    raster = xr.open_rasterio(path)
    raster = raster.drop("band").squeeze("band")
    raster = raster.sel(x=slice(xmin, xmax)).sel(y=slice(ymax, ymin))
    raster = imod.rasterio.resample(
        source=raster,
        like=like,
        src_crs="+init=EPSG:{}".format(4326),
        dst_crs="+init=EPSG:{}".format(epsg),
        reproject_kwargs = {"src_nodata":255}
    )
    return raster

# Get input
builder = model_builder.ModelBuilder()
config = builder.parse_config("imodflow.yaml")
general_config = dict(config[0])
model_config = dict(config[1])
region = hmw.hydro_engine_geometry(general_config["region"], "region")

if model_config["crs"] == "UTM":
    epsg = utm_epsg(region)
else:
    # TODO
    # return proj/crs object
    pass

general_config["hydro-engine"]["defaults"]["crs"] = "EPSG:{}".format(epsg)
proj = pyproj.Proj(init="EPSG:{}".format(epsg))

# Downloads
cli.general_options(region, general_config)
dem = xr.open_rasterio("download/dem.tif")
dem = dem.where(dem != dem.min())
dem = dem.drop("band").squeeze("band")
dem = dem.astype(np.float64)

# Local data
landpath = general_config["local"]["datasets"]["land"]
landmask = get_mask(landpath, region, dem, 0, proj)
dem = dem.where(landmask)
landmask = landmask.where(dem > 0.0)
xmin, ymin, xmax, ymax = sg.shape(region).bounds

# Model generation
## Layer thickness
sediment_dz = load_raster(general_config["local"]["datasets"]["sediment-thickness"], dem, region, epsg)
sediment_dz = sediment_dz.where(sediment_dz != sediment_dz.max()).where(sediment_dz != 0.0)

soil_dz = load_raster(general_config["local"]["datasets"]["soilgrids-depth"], dem, region, epsg)
soil_dz = soil_dz.where(soil_dz != soil_dz.min()).where(soil_dz != 0.0)
soil_dz = soil_dz / 100.0

total_thickness = xr.concat([soil_dz, sediment_dz], dim="layer").max("layer")
L2_dz = total_thickness - soil_dz
L2_dz = L2_dz.where(L2_dz > 0)

## BND definition
botL2 = dem - total_thickness
bndL1 = xr.full_like(botL2, 1).where(~np.isnan(botL2))
bndL1.fillna(0)
bndL1 = bndL1.assign_coords(layer=1)
bndL2 = bndL1.copy().assign_coords(layer=2)
bnd = xr.concat([bndL1, bndL2], dim="layer")

## Tops & Bots
top = bnd.copy()
bot = bnd.copy()
top.sel(layer=1)[...] = dem.where(bnd.sel(layer=1) == 1)
bot.sel(layer=1)[...] = top.sel(layer=1) - soil_dz
bot.sel(layer=2)[...] = top.sel(layer=1) - total_thickness
top.sel(layer=2)[...] = bot.sel(layer=1) - 0.25 * L2_dz

##  Other Packages
shd = bnd.copy()
shd.sel(layer = 1)[...] = top.sel(layer=1)
shd.sel(layer = 2)[...] = top.sel(layer=1)

khv = bnd * 5.0
khv.sel(layer=2)[...] = khv.sel(layer=2) * 5.0
kvv = xr.full_like(bnd.sel(layer=1), 0.005).where(bnd.sel(layer=1) == 1)

drn_bot = top.sel(layer=1).where(bnd.sel(layer=1) == 1)
drn_cond = xr.full_like(drn_bot, 1000.0).where(bnd.sel(layer=1) == 1)

rch = xr.full_like(bnd.sel(layer=1), 1.0).where(bnd.sel(layer=1) == 1)

# Write model
model = OrderedDict()
model["bnd"] = bnd
model["shd"] = shd
model["khv"] = khv
model["kvv"] = kvv
model["top"] = top
model["bot"] = bot
model["rch"] = rch
model["shd"] = shd
model["drn-cond"] = drn_cond
model["drn-bot"] = drn_bot

imod.write("test", model)
