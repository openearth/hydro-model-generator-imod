import os
import shutil
import yaml
import zipfile
import hydroengine
import numpy as np
import logging

from pathlib import Path
from shapely import geometry as sg

import model_generator
from hydro_model_builder import model_builder

logger = logging.getLogger(__name__)


def parse_config(configfile):
    with open(configfile) as f:
        return list(yaml.safe_load_all(f))


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


def get_hydro_data(region, ds):
    logging.info(ds["path"] + " \n  \n  " + ds["variable"] + "    ")
    if ds["function"] == "get-raster" and ds["source"] == "earth-engine":
        # create directory
        Path(ds["path"]).parent.mkdir(parents=True, exist_ok=True)
        if ds["crs"].lower() == "utm":
            ds["crs"] = "EPSG:{}".format(utm_epsg(region))
        hydroengine.download_raster(
            region,
            ds["path"],
            ds["variable"],
            ds["cell_size"],
            ds["crs"],
            ds["region_filter"],
            ds["catchment_level"],
        )
    elif ds["function"] == "get-catchments" and ds["source"] == "earth-engine":
        hydroengine.download_catchments(region, ds["path"], ds["region_filter"], ds["catchment_level"])
    elif ds["function"] == "get-rivers" and ds["source"] == "earth-engine":
        filter_upstream_gt = 1000
        hydroengine.download_rivers(region, ds["path"], filter_upstream_gt, ds["region_filter"], ds["catchment_level"])


def general_options(d):
    # get data from hydro-engine one by one
    defaults = d["hydro-engine"]["defaults"]
    for ds_override in d["hydro-engine"]["datasets"]:
        ds = defaults.copy()
        ds.update(ds_override)
        print("=> general_options.hydro-engine.datasets.variable:", ds["variable"])
        if ds["source"] == "earth-engine":
            get_hydro_data(d["region"], ds)
        else:
            # TODO support non earth-engine datasets
            print("skipped variable:", ds["variable"])


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


def zip_model_output(input_dir, output_dir):
    if os.path.exists(output_dir):
        os.remove(output_dir)
    else:
        print("Can not delete {}".format(output_dir))
    zipf = zipfile.ZipFile(output_dir, 'w', zipfile.ZIP_DEFLATED)
    zipdir(input_dir, zipf)
    zipf.close()
    return zipf

def delete_output_files(input_dir):
    shutil.rmtree(input_dir)
    
    origfolder = "hydro-engine/"
    for item in os.listdir(origfolder):
        if item.endswith(".tif"):
            os.remove(os.path.join(origfolder, item))
        if item.endswith(".tfw"):
            os.remove(os.path.join(origfolder, item))
        if item.endswith(".json"):
            os.remove(os.path.join(origfolder, item))
    # pass

def main():
    path = 'imodflow_onlyHE.yaml'
    # parse yaml (hydro_model_builder.parse_config)
    dicts = parse_config(path)
    genopt, modopt = dicts
    # get hydro data (hydro_model_builder.general_options)
    general_options(genopt)
    # run genwf = ModelGeneratorWflow() genwf.generate_model(genopt, modopt)
    # genwf = ModelGeneratorWflow()
    # genwf.generate_model(genopt, modopt)
    model_builder.fetch_data(genopt)
    model_generator.build_model(**modopt, general_options=genopt)

    # TODO: add relative paths
    input_dir = modopt['modelname']
    output_dir = modopt['modelname'] + '.zip'
    # output_dir = "{0}-{1}.zip".format(model['type'], model['id'])

    zipped_file = zip_model_output(input_dir, output_dir)

    # clean up files not used
    delete_output_files(input_dir)


# hydro_model_builder needs to specify path where templates are,
# what file to save output file to,

if __name__ == '__main__':
    main()
