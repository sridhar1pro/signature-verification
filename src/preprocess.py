import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte

def keep_largest_component(bin_img):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    if num_labels <= 1:
        return bin_img
    # pick component with largest area except background
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest_label).astype("uint8") * 255
    return mask

def preprocess_image(path, target_height=300):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    # Resize uniformly
    h = img.shape[0]
    scale = target_height / float(h)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ----- NEW: background normalization -----
    blur_bg = cv2.GaussianBlur(gray, (55,55), 0)
    norm = cv2.divide(gray, blur_bg, scale=255)

    # ----- NEW: adaptive threshold (best for paper noise) -----
    th = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,    # block size (tune)
        10     # bias (tune)
    )

    # ----- NEW: remove small speckles -----
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    # ----- VERY IMPORTANT -----
    # Keep ONLY the biggest connected component (the signature strokes)
    clean = keep_largest_component(clean)

    # ----- skeletonization -----
    bw = clean > 0
    skel = skeletonize(bw)   # boolean
    skel_u8 = img_as_ubyte(skel)

    return skel_u8
