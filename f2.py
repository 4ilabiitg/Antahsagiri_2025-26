# combo of filters

import cv2
import numpy as np
import os

# -----------------------------
# Enhancement Filters
# -----------------------------

def white_balance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    avg_a = np.average(lab[:, :, 1])
    avg_b = np.average(lab[:, :, 2])

    # Adjust a and b channels
    lab[:, :, 1] -= ((avg_a - 128.0) * (lab[:, :, 0] / 255.0) * 1.1)
    lab[:, :, 2] -= ((avg_b - 128.0) * (lab[:, :, 0] / 255.0) * 1.1)

    # Clip and convert back to uint8
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def gamma_correction(img, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def unsharp_mask(img, strength=1.5):
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + strength, blur, -strength, 0)

def denoise_bilateral(img):
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# -----------------------------
# Analysis Filters
# -----------------------------

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

def median_blur(img, ksize=5):
    """Remove salt-and-pepper noise."""
    return cv2.medianBlur(img, ksize)

def histogram_equalization_gray(img):
    """Apply histogram equalization to grayscale version of image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized

def gabor_filter(img):
    """Highlight textures using a Gabor filter."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
    return filtered

def sobel_edges(img):
    """Highlight horizontal and vertical edges using Sobel operator."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    return cv2.convertScaleAbs(sobel_combined)

def morphological_operations(img):
    """Apply dilation and erosion to highlight or remove features."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=1)
    eroded = cv2.erode(gray, kernel, iterations=1)
    return dilated, eroded

def color_inversion(img):
    """Invert image colors."""
    return cv2.bitwise_not(img)

def adaptive_thresholding(img):
    """Highlight local features using adaptive thresholding."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=11, C=2
    )
    return thresh

# -----------------------------
# Main Pipeline
# -----------------------------

def process_underwater_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to load image.")

    # Enhancement pipeline
    enhanced = white_balance(img)
    enhanced = apply_clahe(enhanced)
    enhanced = gamma_correction(enhanced, gamma=1.2)
    enhanced = denoise_bilateral(enhanced)
    enhanced = unsharp_mask(enhanced)
    
    median = median_blur(enhanced)
    hist_eq = histogram_equalization_gray(enhanced)
    gabor = gabor_filter(enhanced)
    sobel = sobel_edges(enhanced)
    morph_dilate, morph_erode = morphological_operations(enhanced)
    inverted = color_inversion(enhanced)
    adaptive_thresh = adaptive_thresholding(enhanced)

    # Display additional filters
    cv2.imshow("Median Blur", median)
    cv2.imshow("Histogram Equalization", hist_eq)
    cv2.imshow("Gabor Filter", gabor)
    cv2.imshow("Sobel Edges", sobel)
    cv2.imshow("Morph - Dilation", morph_dilate)
    cv2.imshow("Morph - Erosion", morph_erode)
    cv2.imshow("Color Inversion", inverted)
    cv2.imshow("Adaptive Thresholding", adaptive_thresh)

    # Edge detection
    laplacian_edges = edge_detection_laplacian(enhanced)
    canny_edges = edge_detection_canny(enhanced)

    # Object detection
    contours_img = object_detection_contours(enhanced)

    # Display results
    # cv2.imshow("Original", img)
    # cv2.imshow("Enhanced", enhanced)
    # cv2.imshow("Laplacian Edges", laplacian_edges)
    # cv2.imshow("Canny Edges", canny_edges)
    # cv2.imshow("Contours", contours_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -----------------------------
# Run the script
# -----------------------------

if __name__ == "__main__":
    image_path = r"C:\Users\isami\OneDrive\Attachments\Desktop\antah filters\129_img_.png"
    process_underwater_image(image_path)
