# -- coding: utf-8 --
"""
Created on Sun Apr 13 15:24:21 2025
@author: HP
"""

import cv2
import numpy as np
import os

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def hsv_stretching(image, hue_shift=0, sat_scale=1.0, val_scale=1.0):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    h = (h + hue_shift) % 180
    s = np.clip(s * sat_scale, 0, 255).astype(np.uint8)
    v = np.clip(v * val_scale, 0, 255).astype(np.uint8)
    stretched_hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(stretched_hsv, cv2.COLOR_HSV2BGR)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def unsharp_mask(image, strength=3):
    blur = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1 + strength, blur, -strength, 0)

def gray_world(img):
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3
    img[:, :, 0] = np.clip(img[:, :, 0] * (avg_gray / avg_b), 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * (avg_gray / avg_g), 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * (avg_gray / avg_r), 0, 255)
    return img.astype(np.uint8)

def gamma_correction(img, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)



# ----------------------------
# Main Code
# ----------------------------
if __name__ == "__main__":

    # Replace with your full image path
    image_path = r"C:\Users\isami\OneDrive\Attachments\Desktop\antah filters\19_img_.png"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    input_image = cv2.imread(image_path)

    if input_image is None:
        raise ValueError("Failed to load the image. Check the file format or path.")
    
        # === Extra filters for exploration ===
 


    # Apply filters
    #processed_image = white_balance(input_image)
    #processed_image = apply_clahe(input_image)
    #processed_image = gamma_correction(input_image, gamma=1.2)
    #processed_image = unsharp_mask(input_image)
    processed_image =hsv_stretching(input_image, hue_shift=0, sat_scale=1.0, val_scale=1.0)

    # Show the images
    cv2.imshow("Original Image", input_image)
    cv2.imshow("Enhanced Image", processed_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
# -- coding: utf-8 --
"""
Created on Sun Apr 13 15:24:21 2025
@author: HP
"""

import cv2
import numpy as np
import os

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def hsv_stretching(image, hue_shift=0, sat_scale=1.0, val_scale=1.0):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    h = (h + hue_shift) % 180
    s = np.clip(s * sat_scale, 0, 255).astype(np.uint8)
    v = np.clip(v * val_scale, 0, 255).astype(np.uint8)
    stretched_hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(stretched_hsv, cv2.COLOR_HSV2BGR)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def unsharp_mask(image, strength=1.5):
    blur = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1 + strength, blur, -strength, 0)

def gray_world(img):
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3
    img[:, :, 0] = np.clip(img[:, :, 0] * (avg_gray / avg_b), 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * (avg_gray / avg_g), 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * (avg_gray / avg_r), 0, 255)
    return img.astype(np.uint8)

def gamma_correction(img, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def denoise_bilateral(img):
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

def edge_detection_laplacian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)

def edge_detection_canny(img, low_thresh=50, high_thresh=150):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    return edges

def object_detection_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    return contour_img

# ----------------------------
# Main Code
# ----------------------------
if __name__ == "__main__":

    # Replace with your full image path
    image_path = r"C:\Users\isami\OneDrive\Attachments\Desktop\antah filters\19_img_.png"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    input_image = cv2.imread(image_path)

    if input_image is None:
        raise ValueError("Failed to load the image. Check the file format or path.")

    # Apply filters
    processed_image = white_balance(input_image)
    processed_image = apply_clahe(processed_image)
    processed_image = gamma_correction(processed_image, gamma=1.2)
    processed_image = unsharp_mask(processed_image)

    # Show the images
    cv2.imshow("Original Image", input_image)
    cv2.imshow("Enhanced Image", processed_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
