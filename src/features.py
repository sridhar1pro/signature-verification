# src/features.py
import cv2
import numpy as np

def resample_curve(pts, n_points=200):
    # pts: (N,2) float
    # compute cumulative arc length
    diffs = np.diff(pts, axis=0)
    seglen = np.hypot(diffs[:,0], diffs[:,1])
    cum = np.concatenate([[0], np.cumsum(seglen)])
    if cum[-1] == 0:
        return np.repeat(pts[:1], n_points, axis=0)
    # desired distances
    distances = np.linspace(0, cum[-1], n_points)
    resampled = []
    j = 0
    for d in distances:
        while j < len(cum)-1 and cum[j+1] < d:
            j += 1
        if cum[j+1] == cum[j]:
            resampled.append(pts[j])
        else:
            t = (d - cum[j]) / (cum[j+1] - cum[j])
            p = (1-t) * pts[j] + t * pts[j+1]
            resampled.append(p)
    return np.array(resampled)

def contour_to_points(bin_img, n_points=200):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    # pick largest contour by area or length (prefer length)
    contour = max(contours, key=lambda c: c.shape[0])
    pts = contour.squeeze().astype(np.float32)  # (N,2)

    # ensure 2D
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]

    # resample to fixed length for stability
    pts_res = resample_curve(pts, n_points=n_points)

    # normalize coordinates to bounding box
    x_min, y_min = pts_res.min(axis=0)
    x_max, y_max = pts_res.max(axis=0)
    w = x_max - x_min if (x_max - x_min) > 1e-6 else 1.0
    h = y_max - y_min if (y_max - y_min) > 1e-6 else 1.0
    norm_xy = (pts_res - [x_min, y_min]) / [w, h]

    # compute angles (tangent)
    dx = np.gradient(pts_res[:,0])
    dy = np.gradient(pts_res[:,1])
    angles = np.arctan2(dy, dx)
    angles = angles.reshape(-1,1)

    features = np.hstack([norm_xy, angles])  # shape (n_points, 3)
    return features
