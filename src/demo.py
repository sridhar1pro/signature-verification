# src/demo.py (clean + debug + stable)
from preprocess import preprocess_image
from features import contour_to_points
from greedy import greedy_score
from dtw_dp import dtw_distance
import cv2, os

REF_PATH = 'data/reference/0001_ref.jpg'
QUERY_PATH = 'data/query/0001_q1.jpg'
DEBUG_DIR = 'data/debug'
os.makedirs(DEBUG_DIR, exist_ok=True)

def safe_extract(path, tag):
    img = preprocess_image(path)
    # save preprocessed image for inspection
    fname = os.path.join(DEBUG_DIR, f'{tag}_pre.jpg')
    cv2.imwrite(fname, img)

    try:
        feats = contour_to_points(img)
        if feats is None or len(feats) == 0:
            print(f'Warning: no contour points extracted from {path}')
            return None, img
        return feats, img
    except Exception as e:
        print(f'Error extracting features from {path}:', e)
        return None, img

# extract features
A, ref_img = safe_extract(REF_PATH, 'ref')
B, qry_img = safe_extract(QUERY_PATH, 'query')

# debug info
print("Ref features shape:", None if A is None else A.shape)
print("Query features shape:", None if B is None else B.shape)

# if either missing, stop early
if A is None or B is None:
    print("Cannot compute similarity — missing features.")
    cv2.imwrite(os.path.join(DEBUG_DIR, 'ref_query_side_by_side.jpg'),
                cv2.hconcat([ref_img, qry_img]))
    exit(0)

# compute greedy score (for info only)
gscore = greedy_score(A, B)
print("Greedy score:", gscore)

# always run DTW (we fixed resampling so safe)
d = dtw_distance(A, B, band=None)
print("DTW distance:", d)

# final decision threshold
if d < 0.45:
    print("Decision: Genuine")
else:
    print("Decision: Forged")

# Save feature point overlays
def draw_points(img, pts, outpath):
    im = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = im.shape[:2]
    for (x, y, *_) in pts:
        cx = int(x * w)
        cy = int(y * h)
        cv2.circle(im, (cx, cy), 2, (0,255,0), -1)
    cv2.imwrite(outpath, im)

draw_points(ref_img, A, os.path.join(DEBUG_DIR, 'ref_points.jpg'))
draw_points(qry_img, B, os.path.join(DEBUG_DIR, 'query_points.jpg'))

print("Saved feature visualizations to", DEBUG_DIR)

