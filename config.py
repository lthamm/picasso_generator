import os

PROJECT_ROOT = "."  # assume the script is called from within project root
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
ORIGS_PATH = os.path.join(DATASET_ROOT, "fasseg", "heads_original")
NO_FACES_PATH = os.path.join(PROJECT_ROOT, "dataset", "fasseg", "heads_no_face")
LABELS_PATH = os.path.join(PROJECT_ROOT, "dataset", "fasseg", "heads_labels")

PICASSO_ROOT = os.path.join(DATASET_ROOT, 'picasso_dataset')
"""Path to the root directory into which to save the generated images."""

DATASET_SIZE = 2000
CREATE_MASKS = True


FEATURES = ["left_eye", "right_eye", "nose", "mouth"]
