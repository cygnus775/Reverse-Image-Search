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

CONFIG_PATH = os.path.join("configs", "config.json")
configs = read(CONFIG_PATH)
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


# get file names
filenames = pickle.load(open(FILENAMES, "rb"))
feature_list = pickle.load(open(UNCOMPRESSED_FEATURES, "rb"))

# PCA
pca = PCA(n_components=PCA_DIMS)
pca.fit(feature_list)

# load annoy
annoy_index = AnnoyIndex(ANNOY_DIMS, 'angular')
annoy_index.load(ANNOY)

# Create logger
logger = create_logger(logger_name=__name__, log_file="logs/webapp.log")

# Create fastapi app
logger.debug("Creating fastapi app")
app = FastAPI()

# serve static files
app.mount("/static", StaticFiles(directory="data"), name="static")


# Create root route
@app.get("/")
def index():
    return {
        "message": "Hello world"
    }

@app.post("/api/upload_image")
def upload_image(upload_file: UploadFile = File(...), num_images: int = 10):
    path = os.path.join(DATA_RAW, str(upload_file.filename).replace(" ", "_"))
    print(path)
    with open(path, "w+b") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    upload_image_features = extract_features(path, model)
    compressed_upload_image_features = pca.transform(np.expand_dims(upload_image_features, axis=0))
    print(compressed_upload_image_features.shape)

    annoy_search = annoy_index.get_nns_by_vector(compressed_upload_image_features[0], num_images, include_distances=True)

    similar_filenames = [filenames[i] for i in annoy_search[0]]

    filtered_filenames = []
    for i in similar_filenames:
        split_names = i.split("/")
        split_names.pop(0)
        filtered_filenames.append(f'/static/{"/".join(split_names)}')

    print(filtered_filenames)


    return {
        "message": filtered_filenames
    }