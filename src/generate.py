# Import libraries
import numpy as np
import pickle
from tqdm import tqdm
from keras.applications import ResNet50V2
from utils.feature_extraction import generate_annoy
from utils.feature_extraction import compress_features
from utils.feature_extraction import extract_features
from utils.feature_extraction import get_file_list

# Load model
model = ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='avg'
)

root_dir = "../data/raw"
filenames = sorted(get_file_list(root_dir))

# Extract the feature list
feature_list = []
for i in tqdm(range(len(filenames))):
    feature_list.append(extract_features(filenames[i], model))

# Save filenames and features
pickle.dump(feature_list, open("../data/processed/features_uncompressed.pickle", "wb"))
pickle.dump(filenames, open("../data/processed/filenames.pickle", "wb"))

# Load back from pickle file
filenames = np.array(pickle.load(open("../data/processed/filenames.pickle", "rb")))
feature_list = np.array(pickle.load(open("../data/processed/features_uncompressed.pickle", "rb")))

# Perform PCA
feature_list_compressed = compress_features(feature_list)

# Generate and save annoy trees
annoy_index_path = "../data/processed/annoy.ann"
generate_annoy(filenames, feature_list_compressed, annoy_index_path)
