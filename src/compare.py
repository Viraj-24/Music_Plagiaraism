import cv2
import os
import glob
import numpy as np
from skimage.metrics import structural_similarity as ssim

def load_image(image_path):
    """Load an image and convert it to grayscale."""
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return None
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def compare_images(image1_path, image2_path):
    """Compare two images and return their similarity percentage."""
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)

    if img1 is None or img2 is None:
        print(f"Skipping comparison: '{image1_path}' or '{image2_path}' could not be loaded.")
        return None

    # Resize images to the same shape if they are different
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Compute SSIM (Structural Similarity Index)    
    similarity_index = ssim(img1, img2)
    similarity_percentage = similarity_index * 100  # Convert to percentage

    return similarity_percentage

# Get all image files from the directory (PNG, JPG, JPEG)
image_files = glob.glob("output/*.[pjP][nNgG]*")

# Ensure at least two images are available for comparison
if len(image_files) < 2:
    print("Error: Need at least two images to compare.")
else:
    print("Comparing images in 'output/' directory:\n")
    for i in range(len(image_files) - 1):
        img1, img2 = image_files[i], image_files[i + 1]
        similarity = compare_images(img1, img2)
        if similarity is not None:
            print(f"Similarity between '{os.path.basename(img1)}' and '{os.path.basename(img2)}': {similarity:.2f}%")
