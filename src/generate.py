# Import libraries
import os
import logging
import numpy as np
import pickle
from tqdm import tqdm
from keras.applications import ResNet50V2
from src.utils.feature_extraction import generate_annoy, compress_features, extract_features, get_file_list
from src.utils.read_json import read
from src.utils.log_util import create_logger

logger = create_logger(logger_name=__name__, log_file="logs/generate.log")

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

# Load model
logger.debug(f"Loading model {MODEL}")
model = ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='avg'
)
logger.debug(f"Loaded model {MODEL} successfully")

logger.debug("Getting filenames")
filenames = sorted(get_file_list(DATA_RAW))

def generate():
    # Extract the feature list
    logger.debug("Extracting image features")
    feature_list = []
    for i in tqdm(range(len(filenames))):
        feature_list.append(extract_features(filenames[i], model))
    logger.debug("Feature extraction successful")

    # Make 'processed' directory if it does not exist
    logger.debug("Creating processed directory if it does not exist")
    os.makedirs(PROCESSED_DATA, exist_ok=True)

    # Save filenames and features
    try:
        logger.debug(f"saving features to {UNCOMPRESSED_FEATURES} and file names to {FILENAMES}")
        pickle.dump(feature_list, open(UNCOMPRESSED_FEATURES, "wb"))
        pickle.dump(filenames, open(FILENAMES, "wb"))
    except Exception as e:
        logger.exception(f"Failed to save {UNCOMPRESSED_FEATURES} and {FILENAMES}")

    # Load back from pickle file
    try:
        logger.debug(f"Loading {UNCOMPRESSED_FEATURES}")
        feature_list = np.array(pickle.load(open(UNCOMPRESSED_FEATURES, "rb")))
        logger.debug(f"loading {UNCOMPRESSED_FEATURES} successful")
    except Exception as e:
        logger.exception(f"Failed to read {UNCOMPRESSED_FEATURES}")

    # Perform PCA
    try:
        logger.debug("Performing PCA on features")
        feature_list_compressed = compress_features(feature_list, num_dimensions=PCA_DIMS)
        logger.debug("Completed PCA successfully")
    except Exception as e:
        logger.exception("Failed to perform PCA")

    try:
        logger.debug(f"Saving compressed features to {COMPRESSED_FEATURES}")
        pickle.dump(feature_list_compressed, open(COMPRESSED_FEATURES, "wb"))
    except Exception as e:
        logger.exception(f"failed to save compressed features to {COMPRESSED_FEATURES}")


    # Generate and save annoy trees
    try:
        logger.debug(f"Generating annoy index to {ANNOY}")
        generate_annoy(filenames, feature_list_compressed, ANNOY, num_trees=ANNOY_TREES, num_dimensions=ANNOY_DIMS)
        logger.debug(f"Annoy index generated successfully at {ANNOY}")
    except Exception as e:
        logger.exception(f"Failed to generate annoy index to {ANNOY}")

if __name__ == "__main__":
    generate()