import pickle
from pathlib import Path

import cv2
import numpy as np

############################################
#  Basic helpers (minimal, no CLI)         #
############################################

def project_point(X_w: np.ndarray, K: np.ndarray, extr: np.ndarray) -> tuple[float, float]:
    """Transform *world* point → pixel coordinates.

    Parameters
    ----------
    X_w   : (3,)  XYZ in world frame [m]
    K     : (3,3) intrinsics (OpenCV)
    extr  : (3,4) [R|t] world→camera
    """
    X_c = extr @ np.append(X_w, 1.0)  # camera space
    if X_c[2] <= 0:
        raise ValueError("Point behind camera")
    uv_h = K @ X_c
    return float(uv_h[0] / uv_h[2]), float(uv_h[1] / uv_h[2])

############################################
#  Episode I/O                             #
############################################

def load_episode(pkl_path: str | Path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    cam = d["observation"]["head_camera"]
    return (
        cam["rgb"],
        cam["intrinsic_cv"].astype(float),
        cam["extrinsic_cv"].astype(float),
        d["endpose"].astype(float),
    )

############################################
#  Main drawing routine                    #
############################################

def annotate_episode(
    pkl_path: str | Path,
    output: str | Path | None = None,
    show: bool = False,
):
    """Load episode, project end‑effectors, draw red *rings* on image."""

    img_rgb, K, extr, endpose = load_episode(pkl_path)

    ep = endpose.reshape(2, 7)
    pts = {
        "left": project_point(ep[0, :3], K, extr),
        "right": project_point(ep[1, :3], K, extr),
    }

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Draw red hollow circles (radius 10 px, thickness 2 px)
    for label, (u, v) in pts.items():
        center = (int(round(u)), int(round(v)))
        cv2.circle(img_bgr, center, 10, (0, 0, 255), 2)               # outer red ring
        cv2.putText(img_bgr, label, (center[0] + 8, center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if output:
        cv2.imwrite(str(output), img_bgr)
        print(f"Saved → {output}")
    if show or not output:
        cv2.imshow("Annotation", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

############################################
#  Quick usage                             #
############################################

if __name__ == "__main__":
    PKL = Path("/root/data/projects/embodiedai/RoboticsManipulation/RoboticsManipulation/data/dual_bottles_pick_hard_D435_pkl/episode0/100.pkl")
    annotate_episode(PKL, output="test.png", show=False)