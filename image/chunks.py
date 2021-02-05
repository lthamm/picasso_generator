import os
from os import listdir
from typing import List, Tuple

import numpy as np
from skimage import io

def make(config):
    """Read the original files, then for every file extract the pixel values
    and coordinates for eyes, mouth and nose"""

    origs_files = listdir(config.ORIGS_PATH)
    no_faces_files = listdir(config.NO_FACES_PATH)
    labels_files = listdir(config.LABELS_PATH)

    print(f'[*] Read {len(origs_files)} original, {len(no_faces_files)} empty faces & {len(labels_files)} labels')

    # List of tuples (original image, image with face removed, segmentation of orig image)"""
    data_chunks: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # read all files
    for f in origs_files:
        orig_img = io.imread(os.path.join(config.ORIGS_PATH, f))
        no_face_img = io.imread(os.path.join(config.NO_FACES_PATH, f))
        label_img = io.imread(os.path.join(config.LABELS_PATH, f))
        data_chunks.append(tuple((orig_img, no_face_img, label_img)))

    print(f'[+] Got {len(data_chunks)} chunks')

    # prepare extraction
    pixels_per_image = []  # list of pixels for mouth, eyes, nose for each image
    positions_per_image = []  # positions for each organ

    # for every image copy features
    for chunk in data_chunks:

        chunk_pixels = {}
        chunk_positions = {}

        # left eye
        indices = np.where(np.all(chunk[2] == (0, 0, 255), axis=-1))
        pixels = chunk[0][indices]
        min_row = np.min(indices[0])
        min_col = np.min(indices[1])
        chunk_positions['left_eye'] = tuple((min_row, min_col))

        # what does this do?
        for i in range(len(indices[0])):
            indices[0][i] -= min_row
        for i in range(len(indices[1])):
            indices[1][i] -= min_col

        chunk_pixels['left_eye'] = tuple((indices, pixels))

        # right eye
        indices = np.where(np.all(chunk[2] == (255, 0, 255), axis=-1))
        pixels = chunk[0][indices]
        min_row = np.min(indices[0])
        min_col = np.min(indices[1])
        chunk_positions['right_eye'] = tuple((min_row, min_col))

        for i in range(len(indices[0])):
            indices[0][i] -= min_row
        for i in range(len(indices[1])):
            indices[1][i] -= min_col
        chunk_pixels['right_eye'] = tuple((indices, pixels))

        # nose
        indices = np.where(np.all(chunk[2] == (0, 255, 255), axis=-1))
        pixels = chunk[0][indices]
        min_row = np.min(indices[0])
        min_col = np.min(indices[1])
        chunk_positions['nose'] = tuple((min_row, min_col))
        for i in range(len(indices[0])):
            indices[0][i] -= min_row
        for i in range(len(indices[1])):
            indices[1][i] -= min_col
        chunk_pixels['nose'] = tuple((indices, pixels))

        # mouth
        indices = np.where(np.all(chunk[2] == (0, 255, 0), axis=-1))
        pixels = chunk[0][indices]
        min_row = np.min(indices[0])
        min_col = np.min(indices[1])
        chunk_positions['mouth'] = tuple((min_row, min_col))

        for i in range(len(indices[0])):
            indices[0][i] -= min_row
        for i in range(len(indices[1])):
            indices[1][i] -= min_col
        chunk_pixels['mouth'] = tuple((indices, pixels))

        pixels_per_image.append(chunk_pixels)
        positions_per_image.append(chunk_positions)

    print(f'[+] Succesfullly generated {len(data_chunks)} parts')

    return data_chunks, pixels_per_image, positions_per_image
