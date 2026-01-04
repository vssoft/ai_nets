# This file does two things:
# Exposes a clean load_busi_dataset() function
# Allows testing it standalone via python busi_loader.py

import os
import re
from glob import glob
import cv2
import numpy as np


def get_base_id(filepath: str) -> str:
    """
    Extract base ID from BUSI filenames.
    Example:
      benign (4).png
      benign (4)_mask.png
      benign (4)_mask_1.png
    Returns:
      "benign (4)"
    """
    name = os.path.splitext(os.path.basename(filepath))[0]
    name = re.sub(r"_mask.*$", "", name, flags=re.IGNORECASE)
    return name


def is_mask_file(filepath: str) -> bool:
    """True if filename contains '_mask' (case-insensitive)."""
    return "_mask" in os.path.basename(filepath).lower()


def build_image_mask_map(folder_paths):
    """
    mapping: base_id -> {"image": <path>, "masks": [<mask paths>]}
    """
    mapping = {}

    for folder in folder_paths:
        for file_path in glob(os.path.join(folder, "*")):
            if not os.path.isfile(file_path):
                continue

            base_id = get_base_id(file_path)

            if base_id not in mapping:
                mapping[base_id] = {"image": None, "masks": []}

            if is_mask_file(file_path):
                mapping[base_id]["masks"].append(file_path)
            else:
                if mapping[base_id]["image"] is None:
                    mapping[base_id]["image"] = file_path

    return mapping


def load_busi_dataset(folder_paths, size=128, verbose=True):
    """
    Load BUSI dataset robustly.
    Returns:
      X: [N,1,H,W] float32, grayscale normalized
      y: [N,1,H,W] float32, binary masks
    """
    mapping = build_image_mask_map(folder_paths)

    images = []
    masks = []

    missing_image = []
    missing_mask = []

    for base_id, entry in sorted(mapping.items()):
        if entry["image"] is None:
            missing_image.append(base_id)
            continue
        if len(entry["masks"]) == 0:
            missing_mask.append(base_id)
            continue

        # Load image
        img = cv2.imread(entry["image"])
        if img is None:
            continue
        img = cv2.resize(img, (size, size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0

        # Combine masks
        combined_mask = np.zeros((size, size), dtype=np.float32)

        for mp in entry["masks"]:
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m = cv2.resize(m, (size, size))
            m = m.astype(np.float32) / 255.0
            combined_mask += m

        combined_mask = (combined_mask > 0.5).astype(np.float32)

        images.append(img)
        masks.append(combined_mask)

    assert len(images) == len(masks), "Mismatch: number of images and masks"

    X = np.expand_dims(np.array(images, dtype=np.float32), axis=1)  # [N,1,H,W]
    y = np.expand_dims(np.array(masks, dtype=np.float32), axis=1)   # [N,1,H,W]

    if verbose:
        print(f"Loaded {len(X)} samples.")
        if missing_image:
            print(f"WARNING: {len(missing_image)} entries had masks but no image (first 5): {missing_image[:5]}")
        if missing_mask:
            print(f"WARNING: {len(missing_mask)} entries had image but no masks (first 5): {missing_mask[:5]}")
        print("Mask unique values:", np.unique(y))

    return X, y


# Minimal self-test of the loader
if __name__ == "__main__":
    # Randomly selects an image and its corresponding mask to display
    import matplotlib.pyplot as plt
    import numpy as np

    data_path = "D:/pyton/datasets/Dataset_BUSI_with_GT"
    folder_paths = [f"{data_path}/malignant", f"{data_path}/benign"]

    X, y = load_busi_dataset(folder_paths, size=128, verbose=True)

    i = np.random.randint(0, len(X))
    plt.subplot(1, 2, 1); plt.imshow(X[i, 0], cmap="gray"); plt.title("Image")
    plt.subplot(1, 2, 2); plt.imshow(y[i, 0], cmap="gray"); plt.title("Mask")
    plt.show()
