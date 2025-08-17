
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io, filters
import pandas as pd
from sklearn.metrics import pairwise_distances
import os
from sklearn.cluster import KMeans
from PIL import Image

def load_images(folder, num_images=None):
    images = []
    count = 0

    for filename in sorted(os.listdir(folder)):
        if num_images is not None and count >= num_images:
            break
        print(filename)
        img = Image.open(os.path.join(folder, filename))
        img_array = np.array(img)
        images.append(img_array.flatten())
        count += 1

    return np.array(images), sorted(os.listdir(folder))

def main(args):
    image = cv2.imread(args.example_image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape

    data, filenames = load_images(args.data_folder, args.num_images)
    print("The data is loaded")

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(data)
    filename_to_cluster = dict(zip(filenames, cluster_labels))

    for idx, original_filename in enumerate(filename_to_cluster):
        cluster_num = filename_to_cluster[original_filename]
        cluster_folder = os.path.join(args.output_folder, f'cluster_{cluster_num+1}')
        os.makedirs(cluster_folder, exist_ok=True)
        image_path = os.path.join(cluster_folder, original_filename)
        image_reshaped = np.reshape(data[idx], (image_height, image_width, 3))
        Image.fromarray(image_reshaped).save(image_path)

    print("Images assigned to clusters and saved successfully.")

    centroid_folder = args.centroid_folder
    os.makedirs(centroid_folder, exist_ok=True)
    for i, centroid_image in enumerate(kmeans.cluster_centers_):
        centroid_image_reshaped = centroid_image.reshape((image_height, image_width, 3))
        image_filename = os.path.join(centroid_folder, f'centroid_cluster_{i+1}.png')
        Image.fromarray(centroid_image_reshaped.astype(np.uint8)).save(image_filename)

    print("Centroid images saved successfully.")

    image_files = [f for f in os.listdir(centroid_folder) if f.lower().endswith('.png')]
    for image_file in image_files:
        image_path = os.path.join(centroid_folder, image_file)
        image = cv2.imread(image_path)
        pixel_vals = image.reshape((-1, 3))
        pixel_vals = np.float32(pixel_vals)
        mean_pixel_value = np.mean(pixel_vals)
        _, thresholded_image = cv2.threshold(image, mean_pixel_value, 255, cv2.THRESH_TOZERO_INV)
        mask = (thresholded_image > np.min(thresholded_image))
        thresholded_image[mask] = 255
        thresholded_image_gray = cv2.cvtColor(thresholded_image, cv2.COLOR_BGR2GRAY)

        os.makedirs(args.threshold_output_folder, exist_ok=True)
        output_path = os.path.join(args.threshold_output_folder, f'thresholded_{image_file}')
        cv2.imwrite(output_path, thresholded_image_gray)
        plt.subplot(1, 2, 2)
        plt.imshow(thresholded_image_gray, cmap='gray')
        plt.title('Thresholded Image')
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster and threshold images using KMeans.")
    parser.add_argument("--example_image", type=str, required=True, help="Path to a sample image to get dimensions.")
    parser.add_argument("--data_folder", type=str, required=True, help="Folder containing images to cluster.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save clustered images.")
    parser.add_argument("--centroid_folder", type=str, required=True, help="Folder to save centroid images.")
    parser.add_argument("--threshold_output_folder", type=str, required=True, help="Folder to save thresholded centroid images.")
    parser.add_argument("--num_images", type=int, default=50, help="Number of images to load for clustering.")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters for KMeans.")
    args = parser.parse_args()
    main(args)
