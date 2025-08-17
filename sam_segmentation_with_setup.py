
import argparse
import os
import sys
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
# Visualization helpers
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
# Core pipeline
# ----------------------------

def get_white_pixel_coordinates(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        coordinates = []
        for y in range(height):
            for x in range(width):
                if img.getpixel((x, y)) == 255:
                    coordinates.append((x, y))
    return coordinates

def save_mask(mask, score, index, folder_path):
    filename = f"mask_score_index_{index}.npy"
    file_path = os.path.join(folder_path, filename)
    np.save(file_path, mask)

def run_segmentation(args):
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

    white_pixels = get_white_pixel_coordinates(args.threshold_mask_path)
    if len(white_pixels) < args.num_points:
        raise ValueError(f"Not enough white pixels ({len(white_pixels)}) for requested num_points={args.num_points}")
    labels = [1] * len(white_pixels)
    indices = np.random.choice(len(white_pixels), size=args.num_points, replace=False)
    coords = np.array(white_pixels)[indices]
    labels_sampled = np.array(labels)[indices]

    os.makedirs(args.pred_mask_np_folder, exist_ok=True)
    os.makedirs(args.pred_mask_final_folder, exist_ok=True)
    os.makedirs(args.csv_folder, exist_ok=True)

    mask_scores = []
    for filename in sorted(os.listdir(args.cluster_image_folder)):
        if filename.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
            image_path = os.path.join(args.cluster_image_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"[warn] Skipping unreadable file: {image_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            base_filename = os.path.splitext(filename)[0]

            predictor.set_image(image)
            masks, scores, _ = predictor.predict(
                point_coords=coords,
                point_labels=labels_sampled,
                multimask_output=True,
                return_logits=False
            )

            selected_index = 0 if scores[1] > args.score_threshold else 1

            save_mask(masks[selected_index], scores[selected_index], selected_index, args.pred_mask_np_folder)
            mask_scores.append([selected_index, float(scores[selected_index])])
            loaded_mask = np.load(os.path.join(args.pred_mask_np_folder, f"mask_score_index_{selected_index}.npy"))

            binary_mask = (loaded_mask > 0).astype(np.uint8) * 255
            output_filename = f"{base_filename}.png"
            io.imsave(os.path.join(args.pred_mask_final_folder, output_filename), binary_mask)

    with open(os.path.join(args.csv_folder, f"{args.mask_scores_csv}"), mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Score"])
        writer.writerows(mask_scores)

def evaluate_dice(args):
    output_files = sorted([f for f in os.listdir(args.pred_mask_final_folder) if f.lower().endswith(".png")])
    gt_files = sorted([f for f in os.listdir(args.gt_folder) if f.lower().endswith(".png")])

    dice_coefficients = {}
    image_pairs = []

    for gt_file in gt_files:
        try:
            gt_num = int(gt_file.split("_")[-1].split(".")[0])
        except Exception:
            continue
        match_file = next((f for f in output_files if f.split("_")[-1].split(".")[0].isdigit() and
                           int(f.split("_")[-1].split(".")[0]) == gt_num), None)
        if match_file:
            gt_img = cv2.imread(os.path.join(args.gt_folder, gt_file), cv2.IMREAD_GRAYSCALE)
            pred_img = cv2.imread(os.path.join(args.pred_mask_final_folder, match_file), cv2.IMREAD_GRAYSCALE)
            if gt_img is None or pred_img is None:
                print(f"[warn] Skipping pair due to read error: {gt_file}, {match_file}")
                continue
            intersection = np.sum((gt_img == 255) & (pred_img == 255))
            denom = (np.sum(gt_img == 255) + np.sum(pred_img == 255))
            dice = (2.0 * intersection / denom) if denom > 0 else 0.0
            dice_coefficients[(gt_file, match_file)] = dice
            image_pairs.append((gt_file, match_file))
            print("Dice:", dice)
        else:
            print("No match found for", gt_file)

    os.makedirs(args.csv_folder, exist_ok=True)
    with open(os.path.join(args.csv_folder, args.dice_csv), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ground Truth Image", "Predicted Image", "Dice Coefficient"])
        for pair in image_pairs:
            writer.writerow([pair[0], pair[1], dice_coefficients[pair]])

    print("Saved Dice scores to CSV")

def main():
    parser = argparse.ArgumentParser(description="SAM segmentation + Dice with optional setup (install/download).")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="Path to SAM checkpoint .pth file")
    parser.add_argument("--threshold_mask_path", type=str, required=True, help="Path to binary threshold mask (white=255)")
    parser.add_argument("--cluster_image_folder", type=str, required=True, help="Folder containing images to segment")
    parser.add_argument("--pred_mask_np_folder", type=str, required=True, help="Folder to save raw mask .npy files")
    parser.add_argument("--pred_mask_final_folder", type=str, required=True, help="Folder to save final binary masks (.png)")
    parser.add_argument("--csv_folder", type=str, required=True, help="Folder to save CSV outputs")
    parser.add_argument("--gt_folder", type=str, required=True, help="Ground-truth masks folder (.png) for Dice evaluation")
    parser.add_argument("--mask_scores_csv", type=str, default="mask_scores.csv")
    parser.add_argument("--dice_csv", type=str, default="dice_scores.csv")
    parser.add_argument("--num_points", type=int, default=10000)
    parser.add_argument("--score_threshold", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run SAM on")
    parser.add_argument("--model_type", type=str, default="vit_h", help="SAM model type (e.g., vit_h, vit_l, vit_b)")
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
    run_segmentation(args)
    evaluate_dice(args)

if __name__ == "__main__":
    main()
