
import argparse
import os
import sys
import re
import csv
import subprocess
import shutil
import numpy as np
import cv2
from PIL import Image
from skimage import io
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
# ----------------------------
# Optional setup helpers
# ----------------------------

def maybe_pip_install_segment_anything():
    """Try to import segment_anything; if missing, pip install from GitHub."""
    try:
        import segment_anything  # noqa: F401
        return True
    except Exception:
        pass
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "git+https://github.com/facebookresearch/segment-anything.git"])
        import segment_anything  # noqa: F401
        return True
    except Exception as e:
        print("[warn] Failed to install segment-anything from GitHub:", e)
        return False

def maybe_download(url, dst_path):
    """Download file to dst_path if it doesn't exist."""
    if os.path.exists(dst_path):
        return True
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    try:
        if shutil.which("wget"):
            subprocess.check_call(["wget", "-O", dst_path, url])
            return True
        if shutil.which("curl"):
            subprocess.check_call(["curl", "-L", "-o", dst_path, url])
            return True
    except Exception as e:
        print("[warn] External downloader failed, trying urllib:", e)
    try:
        import urllib.request
        urllib.request.urlretrieve(url, dst_path)
        return True
    except Exception as e:
        print("[error] Could not download file:", e)
        return False

# ----------------------------
# Visualization helpers (optional)
# ----------------------------

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(2), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 0/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image, cmap="gray")

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size,
               edgecolor="white", linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size,
               edgecolor="white", linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))

# ----------------------------
# Core helpers
# ----------------------------

def get_white_pixel_coords_from_ndarray(gray_uint8):
    """Return (x,y) coords where pixel value == 255 from a grayscale uint8 array."""
    ys, xs = np.where(gray_uint8 == 255)
    return list(zip(xs.tolist(), ys.tolist()))

def save_mask(mask, score, index, folder_path):
    """Save a mask to a deterministic filename (overwrites across images, like original script)."""
    os.makedirs(folder_path, exist_ok=True)
    filename = f"mask_score_index_{index}.npy"
    file_path = os.path.join(folder_path, filename)
    np.save(file_path, mask)

def extract_trailing_number(name: str):
    """Extract trailing integer before extension; returns None if not found."""
    m = re.search(r'(\d+)(?:\.[^.]+)?$', name)
    return int(m.group(1)) if m else None

def find_gt_for_number(gt_folder, number):
    """Find a GT filename in gt_folder that ends with the same trailing number."""
    for f in os.listdir(gt_folder):
        if f.lower().endswith((".png", ".tif", ".tiff", ".jpg", ".jpeg")):
            n = extract_trailing_number(f)
            if n is not None and n == number:
                return f
    return None

# ----------------------------
# Segmentation + Dice
# ----------------------------

def run_pipeline(args):
    # Optional setup
    if args.auto_install:
        ok = maybe_pip_install_segment_anything()
        if not ok:
            print("[warn] Proceeding without auto-install success. Make sure 'segment-anything' is available.")
    if args.download_checkpoint_if_missing and args.sam_checkpoint_url:
        _ = maybe_download(args.sam_checkpoint_url, args.sam_checkpoint)
    if args.create_images_dir:
        os.makedirs("images", exist_ok=True)



    device = "cuda" if (args.device.lower() == "cuda" and torch.cuda.is_available()) else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    os.makedirs(args.pred_mask_np_folder, exist_ok=True)
    os.makedirs(args.pred_mask_final_folder, exist_ok=True)
    os.makedirs(args.csv_folder, exist_ok=True)

    mask_scores = []

    # Iterate images in image_folder and match GT by trailing number
    for filename in sorted(os.listdir(args.image_folder)):
        if not filename.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
            continue
        number = extract_trailing_number(filename)
        if number is None:
            print(f"[warn] Skipping {filename}: no trailing number.")
            continue

        gt_file = find_gt_for_number(args.gt_folder, number)
        if gt_file is None:
            print(f"[warn] No GT match for {filename}")
            continue

        # Load image and GT
        image_path = os.path.join(args.image_folder, filename)
        gt_path = os.path.join(args.gt_folder, gt_file)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"[warn] Unreadable image: {image_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        GT = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if GT is None:
            print(f"[warn] Unreadable GT: {gt_path}")
            continue

        # Build point prompts from GT white pixels
        white_pixels = get_white_pixel_coords_from_ndarray(GT.astype(np.uint8))
        if len(white_pixels) < args.num_points:
            print(f"[warn] {filename}: only {len(white_pixels)} white pixels, needs {args.num_points}. Skipping.")
            continue

        indices = np.random.choice(len(white_pixels), size=args.num_points, replace=False)
        coords = np.array(white_pixels)[indices]
        labels = np.ones(len(indices), dtype=np.int64)  # all positives

        # SAM predict
        base_filename = os.path.splitext(filename)[0]
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=True,
            return_logits=False
        )

        selected_index = 0 if scores[1] > args.score_threshold else 1
        save_mask(masks[selected_index], scores[selected_index], selected_index, args.pred_mask_np_folder)
        mask_scores.append([base_filename, int(selected_index), float(scores[selected_index])])

        loaded_mask = np.load(os.path.join(args.pred_mask_np_folder, f"mask_score_index_{selected_index}.npy"))
        binary_mask = (loaded_mask > 0).astype(np.uint8) * 255
        io.imsave(os.path.join(args.pred_mask_final_folder, f"{base_filename}.png"), binary_mask)

    # Save mask scores
    with open(os.path.join(args.csv_folder, args.mask_scores_csv), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ImageBase", "SelectedIndex", "Score"])
        writer.writerows(mask_scores)

    # Dice evaluation
    pred_files = sorted([f for f in os.listdir(args.pred_mask_final_folder) if f.lower().endswith(".png")])
    gt_files = sorted([f for f in os.listdir(args.gt_folder) if f.lower().endswith(".png")])

    dice_coefficients = []
    for gt_file in gt_files:
        gn = extract_trailing_number(gt_file)
        if gn is None:
            continue
        # find pred with same number
        match = None
        for pf in pred_files:
            pn = extract_trailing_number(pf)
            if pn is not None and pn == gn:
                match = pf
                break
        if match is None:
            print(f"[warn] No predicted match for GT {gt_file}")
            continue

        gt_img = cv2.imread(os.path.join(args.gt_folder, gt_file), cv2.IMREAD_GRAYSCALE)
        pr_img = cv2.imread(os.path.join(args.pred_mask_final_folder, match), cv2.IMREAD_GRAYSCALE)
        if gt_img is None or pr_img is None:
            print(f"[warn] Read error for pair GT={gt_file}, Pred={match}")
            continue

        intersection = np.sum((gt_img == 255) & (pr_img == 255))
        denom = (np.sum(gt_img == 255) + np.sum(pr_img == 255))
        dice = (2.0 * intersection / denom) if denom > 0 else 0.0
        dice_coefficients.append([gt_file, match, dice])
        print(f"Dice({gt_file}, {match}) = {dice:.4f}")

    with open(os.path.join(args.csv_folder, args.dice_csv), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ground Truth Image", "Predicted Image", "Dice Coefficient"])
        writer.writerows(dice_coefficients)

    print("Saved:", os.path.join(args.csv_folder, args.mask_scores_csv), "and", os.path.join(args.csv_folder, args.dice_csv))

def main():
    parser = argparse.ArgumentParser(
        description="SAM segmentation using GT prompts per-image + Dice evaluation (with optional setup)."
    )
    # Paths
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="Path to SAM checkpoint .pth file")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder with input images")
    parser.add_argument("--gt_folder", type=str, required=True, help="Folder with GT images (same trailing numbers)")
    parser.add_argument("--pred_mask_np_folder", type=str, required=True, help="Folder to save raw mask .npy files")
    parser.add_argument("--pred_mask_final_folder", type=str, required=True, help="Folder to save final binary masks")
    parser.add_argument("--csv_folder", type=str, required=True, help="Folder to save CSV reports")
    # Options
    parser.add_argument("--num_points", type=int, default=10000, help="Number of prompt points sampled from GT white pixels")
    parser.add_argument("--score_threshold", type=float, default=0.9, help="Mask selection threshold")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run SAM on")
    parser.add_argument("--model_type", type=str, default="vit_h", help="SAM model type (vit_h, vit_l, vit_b)")
    parser.add_argument("--mask_scores_csv", type=str, default="mask_scores.csv")
    parser.add_argument("--dice_csv", type=str, default="dice_scores.csv")
    # Setup toggles
    parser.add_argument("--auto_install", action="store_true",
                        help="Attempt to pip install 'segment-anything' from GitHub if missing")
    parser.add_argument("--download_checkpoint_if_missing", action="store_true",
                        help="Download checkpoint from --sam_checkpoint_url if --sam_checkpoint path doesn't exist")
    parser.add_argument("--sam_checkpoint_url", type=str,
                        default="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                        help="URL to download SAM checkpoint from when enabled")
    parser.add_argument("--create_images_dir", action="store_true",
                        help="Create a local 'images' directory (for parity with original script)")

    args = parser.parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()
