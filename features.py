import os
import numpy as np

# import matplotlib.pyplot as plt
# from ultralytics import YOLO
# import seaborn as sns
# import tensorflow as tf
import cv2
import pandas as pd

# from sklearn.cluster import KMeans
import skimage as ski
from sklearn.decomposition import PCA
import glob


# Set the environment variable to suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Set the random seed for reproducibility
np.random.seed(42)

dataset_path = "mango/dataset.v18i.tensorflow/train/_annotations.csv"

train_dataset = pd.read_csv(dataset_path)

# print(train_dataset.head())


# Extract the mango blocks using coordinates in the dataset
def extract_mango_blocks(dataset):
    """Extract mango blocks from the dataset based on coordinates."""
    mango_blocks = []
    for index, row in dataset.iterrows():
        x_min = int(row["xmin"])
        y_min = int(row["ymin"])
        x_max = int(row["xmax"])
        y_max = int(row["ymax"])
        mango_blocks.append((row["filename"], x_min, y_min, x_max, y_max))
    return mango_blocks


mango_blocks = extract_mango_blocks(train_dataset)

mango_blocks_df = pd.DataFrame(
    mango_blocks, columns=["filename", "x_min", "y_min", "x_max", "y_max"]
)

# print(mango_blocks_df.head())


# Cut the mango blocks from the images and save them using the mango_blocks DataFrame
def cut_and_save_mango_blocks(mango_blocks_df, output_dir="mango_blocks"):
    """Cut and save mango blocks from the dataset."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for index, row in mango_blocks_df.iterrows():
        filename = row["filename"]
        x_min = row["x_min"]
        y_min = row["y_min"]
        x_max = row["x_max"]
        y_max = row["y_max"]

        # Load the image
        img_path = f"E:\\DATA SET\\mango\\dataset.v18i.tensorflow\\train\\{filename}"
        img = cv2.imread(img_path)
        # Check if the image was loaded successfully

        if img is None:
            print(f"Image {filename} not found.")
            continue

        # Cut the mango block
        mango_block = img[y_min:y_max, x_min:x_max]

        # Save the mango block
        output_path = os.path.join(output_dir, f"mango_block_{index}.jpg")
        cv2.imwrite(output_path, mango_block)


# cut_and_save_mango_blocks(mango_blocks_df)
# -----------------------------------------------------------------------
"""
🔸 Color Features (most indicative for fruit aging)
- HSV or Lab color histograms

🔸 Texture Features
- Wrinkles → Local Binary Pattern (LBP)

- Mold → Spot detection via entropy or noise maps

- Surface roughness → GLCM (Gray Level Co-occurrence Matrix)

🔸 Shape Features
- Shriveling causes compact shape — use contour area vs. bounding box

- Irregularity → convexity defects

🔸 Additional
- Blurriness (decay may affect camera focus if fruit oozes)

- Surface reflectance (shiny → dull with age)
"""


###----------------------COLOR FEATURES----------------------###
def extract_color_features(image, h_bins=30, s_bins=32):
    """
    Extract color histogram features from an image in HSV space.

    Args:
        image (np.ndarray): Input image in BGR format.
        h_bins (int): Number of bins for the Hue channel.
        s_bins (int): Number of bins for the Saturation channel.

    Returns:
        np.ndarray: Flattened and normalized HSV color histogram.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))

    hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    masked_pixels = hsv[mask > 0]

    avg_hue = np.mean(masked_pixels[:, 0]) if masked_pixels.size > 0 else 0
    avg_sat = np.mean(masked_pixels[:, 1]) if masked_pixels.size > 0 else 0
    avg_val = np.mean(masked_pixels[:, 2]) if masked_pixels.size > 0 else 0

    return hist, avg_hue, avg_sat, avg_val


###-----------------------TEXTURE FEATURES----------------------###
def extract_texture_features(image):
    """Extract texture features using Local Binary Patterns (LBP)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = ski.feature.local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    return hist


def extract_glcm_features(image):
    """Extract texture features using GLCM (Gray Level Co-occurrence Matrix)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = ski.feature.graycomatrix(
        gray, distances=[1], angles=[0], symmetric=True, normed=True
    )
    contrast = ski.feature.graycoprops(glcm, "contrast")[0, 0]
    dissimilarity = ski.feature.graycoprops(glcm, "dissimilarity")[0, 0]
    homogeneity = ski.feature.graycoprops(glcm, "homogeneity")[0, 0]
    energy = ski.feature.graycoprops(glcm, "energy")[0, 0]
    correlation = ski.feature.graycoprops(glcm, "correlation")[0, 0]
    return [contrast, dissimilarity, homogeneity, energy, correlation]


###-----------------------SHAPE FEATURES----------------------###
def extract_shape_features(image):
    """Extract shape features from the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return [0, 0]  # No contours found

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    return [area, perimeter]


### Function to extract all features from an image
def extract_features(image):
    """Extract all features from the image."""
    color_features, avg_h, avg_s, avg_v = extract_color_features(
        image, h_bins=30, s_bins=32
    )  # 960 + 3
    texture_features = extract_texture_features(image)  # 10
    glcm_features = extract_glcm_features(image)  # 5
    shape_features = extract_shape_features(image)  # 2

    # Combine all features into a single array
    features = np.concatenate(
        [
            color_features,
            [avg_h],
            [avg_s],
            [avg_v],
            texture_features,
            glcm_features,
            shape_features,
        ]
    )
    return features


# -----------------------------------------------------------------------
# Step 1: Get all image paths from the mango_blocks directory
image_paths = glob.glob("mango_blocks/*.jpg")

# Step 2: Extract features from each image
feature_list = []
image_names = []  # Optional: track which feature belongs to which image

for path in image_paths:
    image = cv2.imread(path)
    if image is not None:
        features = extract_features(image)
        feature_list.append(features)
        image_names.append(os.path.basename(path))  # e.g., mango_block_0.jpg
    else:
        print(f"Failed to load image: {path}")


h_bins = 30
s_bins = 32
color_feature_names = [f"hsv_{i}" for i in range(h_bins * s_bins)]
color_feature_names += ["avg_h", "avg_s", "avg_v"]
lbp_feature_names = [f"lbp_{i}" for i in range(10)]
glcm_feature_names = [
    "glcm_contrast",
    "glcm_dissimilarity",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
]
other_feature_names = ["shape_area", "shape_perimeter"]
all_feature_names = (
    color_feature_names + lbp_feature_names + glcm_feature_names + other_feature_names
)

# Convert to a DataFrame
feature_array = np.array(feature_list)
features_df = pd.DataFrame(feature_array, columns=all_feature_names)
features_df.insert(0, "image", image_names)  # Optional column with image filenames
features_df.to_csv("mango_features.csv", index=False)
print("Saved features to mango_features.csv")

print(features_df.head())
# -----------------------------------------------------------------------
