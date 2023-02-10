# Load Libraries
import keras.utils as image
from keras.applications.resnet_v2 import preprocess_input
import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
from annoy import AnnoyIndex
import os


# Function to load, preprocess images and extract the features
def extract_features(img_path, model):
    input_shape = (224, 224, 3)
    img = image.load_img(
        img_path,
        target_size=(input_shape[0], input_shape[1]),
    )

    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img, verbose=0)
    flattened_features = features.flatten()
    normalized_features = flattened_features/norm(flattened_features)
    return normalized_features


# Get all file paths of images
def get_file_list(root_dir):
    file_list = []
    counter = 1
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
                counter += 1
    return file_list


# Perform PCA
def compress_features(feature_list, num_dimensions: int = 100):
    pca = PCA(n_components=num_dimensions)
    pca.fit(feature_list)

    # Generate compressed feature list
    return pca.transform(feature_list)


# Generate Annoy index
def generate_annoy(filenames: list, feature_list_compressed: list, save_path: str, num_trees: int = 100,
                   num_dimensions: int = 100):

    # Generate annoy index files
    annoy_index = AnnoyIndex(num_dimensions, metric='angular')
    for i in range(len(filenames)):
        annoy_index.add_item(i, feature_list_compressed[i])
    annoy_index.build(num_trees)
    annoy_index.save(save_path)
