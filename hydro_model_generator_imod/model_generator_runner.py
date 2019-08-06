import os, sys
import shutil
import yaml
import zipfile
import hydroengine
import numpy as np
import logging
import json

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


def get_hydro_data(region, ds, model_id):
    logging.info(ds["path"] + " \n  \n  " + ds["variable"] + "    ")
    ds["path"] = os.path.join(Path(ds["path"]).parent, model_id, Path(ds["path"]).name)
    Path(ds["path"]).parent.mkdir(parents=True, exist_ok=True)
    if ds["function"] == "get-raster" and ds["source"] == "earth-engine":
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
    elif ds["function"] == "get-features":
        download_features(region, ds["source"], ds["path"])
    else:
        raise ValueError(f"Invalid function provided for {ds['variable']}.")

def download_features(region, source, path):
    """Downloads feature collection to JSON file"""
    # TODO remove in due time, when part of hydroengine
    feature_collection = hydroengine.get_feature_collection(region, source)
    with open(path, "w") as f:
        json.dump(feature_collection, f)


def general_options(d, model_id):
    # get data from hydro-engine one by one
    defaults = d["hydro-engine"]["defaults"]
    for ds_override in d["hydro-engine"]["datasets"]:
        ds = defaults.copy()
        ds.update(ds_override)
        print("=> general_options.hydro-engine.datasets.variable:", ds["variable"])
        if ds["source"] == "earth-engine":
            get_hydro_data(d["region"], ds, model_id)
        elif "huite" in ds["source"]:
            get_hydro_data(d["region"], ds, model_id)
        elif "USDOS" in ds["source"]:
            get_hydro_data(d["region"], ds, model_id)
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

def delete_output_files(input_dir, model_id):
    shutil.rmtree(input_dir)
    
    origfolder = "/app/hydro-engine/iMOD/{}/".format(model_id)
    for item in os.listdir(origfolder):
        if item.endswith(".tif"):
            os.remove(os.path.join(origfolder, item))
        if item.endswith(".tfw"):
            os.remove(os.path.join(origfolder, item))
        if item.endswith(".json"):
            os.remove(os.path.join(origfolder, item))
        if item.endswith(".geojson"):
            os.remove(os.path.join(origfolder, item))
    # pass

def main(model_id):
    path = '/app/hydro-generator/yaml/iMOD-{}.yaml'.format(model_id)
    # parse yaml (hydro_model_builder.parse_config)
    dicts = parse_config(path)
    genopt, modopt = dicts
    # get hydro data (hydro_model_builder.general_options)
    general_options(genopt, model_id)
    # run genwf = ModelGeneratorWflow() genwf.generate_model(genopt, modopt)
    # genwf = ModelGeneratorWflow()
    # genwf.generate_model(genopt, modopt)
    # model_builder.fetch_data(genopt)
    model_generator.build_model(**modopt, general_options=genopt, model_id=model_id)

    # TODO: add relative paths
    input_dir = '/app/hydro-input/{}-{}'.format(modopt['modelname'], model_id)
    output_dir = '/app/hydro-input/{}-{}.zip'.format(modopt['modelname'], model_id)
    # output_dir = "{0}-{1}.zip".format(model['type'], model['id'])

    zipped_file = zip_model_output(input_dir, output_dir)

    # clean up files not used
    delete_output_files(input_dir, model_id)


# hydro_model_builder needs to specify path where templates are,
# what file to save output file to,

if __name__ == '__main__':
    main(sys.argv[1])
