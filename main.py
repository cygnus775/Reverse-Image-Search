# import libraries
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from src.utils.log_util import create_logger
import pickle
import shutil
import os
from src.utils.read_json import read
from src.utils.feature_extraction import extract_features
from src.generate import model
from sklearn.decomposition import PCA
from annoy import AnnoyIndex

# Create logger
logger = create_logger(logger_name=__name__, log_file="logs/webapp.log")

# Read configs
logger.debug("Reading config file")
try:
    CONFIG_PATH = os.path.join("configs", "config.json")
    configs = read(CONFIG_PATH)
    logger.debug(f"Read configs successfully from {CONFIG_PATH}")
except Exception as e:
    logger.exception(f"Failed to read configs file {CONFIG_PATH}")

try:
    logger.debug("Reading config values")
    # Data directories
    DATA_ROOT = configs["dirs"]["data_root"]
    DATA_RAW = configs["dirs"]["data_raw"]
    PROCESSED_DATA = configs["dirs"]["processed_data"]

    # Files
    UNCOMPRESSED_FEATURES = configs["files"]["uncompressed_features"]
    COMPRESSED_FEATURES = configs["files"]["compressed_features"]
    FILENAMES = configs["files"]["filenames"]
    ANNOY = configs["files"]["annoy"]

    # Parameters
    MODEL = configs["model"]
    PCA_DIMS = configs["pca"]["num_dimensions"]
    ANNOY_TREES = configs["annoy"]["num_trees"]
    ANNOY_DIMS = configs["annoy"]["num_dimensions"]
except Exception as e:
    logger.exception("Failed to load parameters from config file")

# get file names
logger.debug("Getting filenames and uncompressed features")
filenames = pickle.load(open(FILENAMES, "rb"))
feature_list = pickle.load(open(UNCOMPRESSED_FEATURES, "rb"))

# PCA
logger.debug(f"Performing PCA with {PCA_DIMS} components")
try:
    pca = PCA(n_components=PCA_DIMS)
    logger.debug("Fitting PCA to featurelist")
    pca.fit(feature_list)
except Exception as e:
    logger.exception("Performing PCA Failed")

# load annoy
logger.debug(f"Initializing Annoy with {ANNOY_DIMS} annoy dimensions")
annoy_index = AnnoyIndex(ANNOY_DIMS, 'angular')
try:
    logger.debug(f"Loading Annoy index file from {ANNOY}")
    annoy_index.load(ANNOY)
except Exception as e:
    logger.exception(f"Loading {ANNOY} failed")

# Create fastapi app
logger.debug("Creating fastapi app")
app = FastAPI()

# serve static files
logger.debug(f"Mounting static files to app from {DATA_RAW} to /static")
app.mount("/static", StaticFiles(directory=DATA_RAW), name="static")


# Create root route
@app.get("/")
def index():
    logger.debug("Entered root endpoint")
    return {
        "message": "Hello world"
    }


@app.post("/api/upload_image")
async def upload_image(upload_file: UploadFile = File(...), num_images: int = 10):
    logger.debug(f"Entered /api/upload_image endpoint with file {upload_file.filename} and {num_images} image request")
    path = os.path.join(DATA_RAW, str(upload_file.filename).replace(" ", "_"))
    logger.debug(f"Set file saving path as {path}")

    logger.debug(f"saving image to file at {path}")
    try:
        with open(path, "w+b") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        logger.debug("saving image to file successful")
    except Exception as e:
        logger.exception(f"failed to save {path}")

    logger.debug("Performing feature extraction")
    upload_image_features = extract_features(path, model)
    logger.debug("compressing features")
    compressed_upload_image_features = pca.transform(np.expand_dims(upload_image_features, axis=0))
    logger.debug(f"compressed feature shape = {compressed_upload_image_features.shape}")

    logger.debug("Using annoy to search for nearest neighbours")
    annoy_search = annoy_index.get_nns_by_vector(compressed_upload_image_features[0], num_images,
                                                 include_distances=False)
    logger.debug(f"Annoy search output - {annoy_search}")

    logger.debug("extracting filenames from annoy result")
    similar_filenames = [filenames[i] for i in annoy_search]

    logger.debug("modifying filenames to be available at /static")
    filtered_filenames = []
    for i in similar_filenames:
        split_names = i.split("/")
        split_names.pop(0)
        split_names.pop(0)
        filtered_filenames.append(f'/static/{"/".join(split_names)}')

    logger.debug(f"returning {filtered_filenames} as response")
    return {
        "message": filtered_filenames
    }
