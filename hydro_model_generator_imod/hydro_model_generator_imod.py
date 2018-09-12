import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import geojson
import geopandas as gpd
import imod
import numpy as np
import pyproj
import rasterio
import shapely
import shapely.geometry as sg
import scipy.interpolate
import scipy.ndimage
import scipy.spatial
import xarray as xr

from collections import OrderedDict
from hydro_model_builder.model_generator import ModelGenerator


def get_paths(general_options):
    # TODO: use schema to validate here?
    d = {}
    if "hydro-engine" in general_options:
        for var in general_options["hydro-engine"]["datasets"]:
            d[var["variable"]] = var["path"]
    if "local" in general_options:
        for var in general_options["local"]["datasets"]:
            d[var["variable"]] = var["path"]
    return d


class ModelGeneratorImodflow(ModelGenerator):
    def __init__(self):
        super(ModelGeneratorImodflow, self).__init__()

    def generate_model(self, general_options, options):

        build_model(
            general_options["region"],
            options["cellsize"],
            options["modelname"],
            options["steady_transient"],
            general_options,
        )


def build_model(geojson_path, cellsize, name, steady_transient, general_options):
    """
    Builds and spits out an iMODFLOW groundwater model.
    """
    paths = get_paths(general_options)

    dem = xr.open_rasterio(paths["dem"]).squeeze("band")
    soilgrids_thickness = xr.open_rasterio(paths["soilgrids-thickness"]).squeeze("band")
    sediment_thickness = xr.open_rasterio(paths["sediment-thickness"]).squeeze("band")
    ate_points = gpd.read_file(paths["ate-thickness"])
    ate_mask = gpd.read_file(paths["ate-mask"])
    river_lines = gpd.read_file(paths["rivers"])
    land_polygons = gpd.read_file(paths["land"])
    region = sg.Polygon(geojson.read(geojson_path))
    
    land = rasterize(
        shapes=land_polygons.geometry,
        like=dem,
    )
    is_land = ~np.isnan(land)

    region_raster = rasterize(
        shapes=region,
        like=dem,
    )
    is_region = ~np.isnan(region_raster)

    ate_grid = ate_points_to_grid(ate_points, ate_mask, "sed_thickness_1", dem)
    top, bot = tops_bottoms(dem, soilgrids_thickness, sediment_thickness, ate_grid)

    bnd = xr.full_like(top, 1.0)
    bnd = bnd.where(~np.isnan(top))
    bnd = bnd.where(is_land)
    bnd = bnd.where(is_region)
    is_bnd = bnd.where(~np.isnan(bnd))

    conductivity_polygons = gpd.read_file(paths["aquifer-conductivity"])
    khv, kvv = conductivity(conductivity_polygons, "logK_Ice_x", bnd)

    drn_bot, drn_cond = drainage(dem, cellsize)

    rch = recharge(like=bnd.sel(layer=1))

    shd = starting_head(dem, like=bnd)

    riv_cond, riv_stage, riv_bot, riv_inff = rivers(
        river_lines, "a_WIDTH", "a_DEPTH", dem
    )

    # gather model data
    model = OrderedDict()
    model["bnd"] = bnd
    model["shd"] = shd
    model["khv"] = khv
    model["kvv"] = kvv
    model["top"] = top
    model["bot"] = bot
    model["rch"] = rch
    model["shd"] = shd
    model["drn-bot"] = drn_bot
    model["drn-cond"] = drn_cond
    model["riv-cond"] = riv_cond
    model["riv-stage"] = riv_stage
    model["riv-bot"] = riv_bot
    model["riv-inff"] = riv_inff

    # discard all values outside of bnd
    for k, v in model.items():
        model[k] = v.where(is_bnd)

    imod.write(path=name, model=model, name=name)


def rasterize(shapes, like, fill=np.nan, kwargs={}):
    """
    Rasterize a list of (geometry, fill_value) tuples onto the given
    xarray coordinates.
    """

    raster = rasterio.features.rasterize(
        shapes,
        out_shape=like.shape,
        fill=fill,
        transform=imod.util.transform(like),
        **kwargs,
    )

    return xr.DataArray(raster, like.coords, like.dims)


def ate_points_to_grid(ate_points, ate_area, column, like):
    """
    Smoothes ate_estimate using convolution, triangular kernel, 
    with 20% of total extent.

    Parameters
    ----------
    ate_points : geopandas.GeoDataFrame
        Aquifer Thickness Estimate from Zamrsky et al.

    Returns 
    -------
    xarray.DataArray
    """
    xs = ate_points.geometry.x
    ys = ate_points.geometry.y
    thickness = ate_points[column].values

    xx, yy = np.meshgrid(like.x.values, like.y.values)
    interpolated = scipy.interpolate.griddata(
        points=(xs, ys), values=thickness, xi=(xx, yy), method="nearest"
    )

    triangle_size = int(min(like.x.size, like.y.size) / 5.0)
    if triangle_size % 2 == 0:
        triangle_size += 1
    triangle = scipy.signal.triang(triangle_size)
    kernel_x, kernel_y = np.meshgrid(triangle, triangle)
    kernel = (kernel_x + kernel_y) / 2.0
    smoothed = scipy.ndimage.convolve(interpolated, kernel / kernel.sum())
    ate_valid = rasterize(ate_area.geometry, like)
    smoothed_grid = smoothed.where(ate_valid)
    return smoothed_grid


def tops_bottoms(dem, soilgrids_thickness, sediment_thickness, ate_thickness):
    """
    Creates two aquifer layer geometry (extent, thickness) from parameters.

    Top aquifer is assumed 25% of total thickness
    First resistant layer is assumed 15% of total thickness
    Second aquifer is assumed 60% of total thickness

    Parameters
    ----------
    dem : xarray.DataArray 
        digital elevation model
    soilgrids_thickness : xarray.DataArray 
        thickness from soilgrids estimation
    sediment_thickness : xarray.DataArray 
        thickness from NASA estimation
    ate_thickness : xarray.DataArray
        Aquifer Thickness Estimate, gridded
    
    Returns
    -------
    top : xr.DataArray
    bot : xr.DataArray
    """

    thickness = xr.concat(
        [ate_thickness, soilgrids_thickness, sediment_thickness], dim="layer"
    ).max("layer")
    topL1 = dem.copy()
    topL1 = topL1.assign_coords(layer=1)
    topL2 = topL1 - 0.4 * thickness
    topL2 = topL2.assign_coords(layer=2)
    top = xr.concat([topL1, topL2], dim="layer")

    bot = top.copy()
    bot.sel(layer=1)[...] = top.sel(layer=1) - 0.25 * thickness
    bot.sel(layer=2)[...] = top.sel(layer=1) - thickness

    return top, bot


def conductivity(conductivity_polygons, column, like):
    # from log m/s to m/d
    conductivity_polygons["khv"] = 10 ** (conductivity_polygons[column] / 100) * 1.0e7
    khv = xr.full_like(like, 10.0)
    khv.sel(layer=1)[...] = rasterize(
        shapes=(
            (shape, value)
            for shape, value in zip(
                conductivity_polygons.geometry, conductivity_polygons["khv"]
            )
        ),
        like=like,
    )
    kvv = 0.3 * khv.sel(layer=1) 
    return khv, kvv


def drainage(dem, cellsize):
    drn_bot = dem.copy()
    # assumes drain resistance of 10 days
    drn_cond = xr.full_like(drn_bot, 1.0) * (cellsize * cellsize) / 10.0
    return drn_bot, drn_cond


def recharge(like):
    # assumes 1.0 mm/d recharge
    rch = xr.full_like(like, 1.0)
    return rch


def starting_head(dem, like):
    shd = xr.full_like(like, dem)
    return shd


def raster_to_features(raster):
    """
    Parameters
    ----------
    raster : xarray.DataArray
        containing coordinates x and y, uniformly spaced.
    """
    # generate shapes of cells to use for intersection
    dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(raster)
    a_dx = abs(dx)
    a_dy = abs(dy)
    # "start" corners
    xs = np.arange(xmin, xmax, a_dx)
    ys = np.arange(ymin, ymax, a_dy)
    yy_s, xx_s = np.meshgrid(ys, xs)
    # "end" corners
    xe = np.arange(xmin + dx, xmax + dy, a_dx)
    ye = np.arange(ymin + dy, ymax + dy, a_dy)
    yy_e, xx_e = np.meshgrid(ye, xe)

    A = xx_s.flatten()
    B = xx_e.flatten()
    C = yy_s.flatten()
    D = yy_e.flatten()
    squares = [
        sg.Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        for x1, x2, y1, y2 in zip(A, B, C, D)
    ]
    features = gpd.GeoDataFrame()
    features.geometry = squares
    return features


def coordinate_index(x, xmin, xmax, dx):
    """
    Takes care of reversed coordinates, typically y with a negative value for dy.
    """
    if dx < 0:
        xi = ((xmax - x) / abs(dx)).astype(int)
    else:
        xi = ((x - xmin) / abs(dx)).astype(int)
    return xi


def rivers(rivers_lines, width_column, depth_column, dem):
    # TODO check for model tops and bots
    buffered = []
    for _, row in rivers_lines.iterrows():
        width = row[width_column]
        row = row.geometry.buffer(width / 2.0)
        buffered.append(row)
    rivers_polygons = gpd.GeoDataFrame(buffered)

    # intersection
    gridshape = raster_to_features(dem)
    river_cells = gpd.overlay(rivers_polygons, gridshape, how="intersection")

    centroids = gpd.GeoDataFrame()
    centroids.geometry = river_cells.centroid
    centroids["x"] = centroids.geometry.x
    centroids["y"] = centroids.geometry.y
    centroids["area"] = river_cells.area
    centroids["depth"] = river_cells[depth_column]
    # calculate indices in grid out
    dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(dem)
    centroids["xi"] = coordinate_index(centroids["x"].values, xmin, xmax, dx)
    centroids["yi"] = coordinate_index(centroids["y"].values, ymin, ymax, dy)

    # fill in outgoing grids
    nrow, ncol = dem.x.size, dem.y.size
    area = np.full((nrow, ncol), 0.0)
    # for area weighted depth
    depth_x_area = np.full((nrow, ncol), 0.0)
    for i, j, a, d in zip(
        centroids["yi"], centroids["xi"], centroids["area"], centroids["depths"]
    ):
        area[i, j] += a
        depth_x_area[i, j] += a * d
    depth = depth_x_area / area

    river_resistance = 100.0  # TODO
    conductance = xr.full_like(dem, area / river_resistance)
    is_river = conductance > 0.0
    depth_da = xr.full_like(dem, depth)
    stage = dem - 0.15 * depth_da  # TODO
    bottom = dem - depth_da
    infiltration_factor = xr.full_like(dem, 1.0)

    conductance = conductance.where(is_river)
    stage = stage.where(is_river)
    bottom = bottom.where(is_river)
    infiltration_factor = infiltration_factor.where(is_river)

    return conductance, stage, bottom, infiltration_factor

