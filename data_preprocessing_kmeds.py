
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn_extra.cluster import KMedoids
from PIL import Image

def load_images(folder, num_images=None):
    images = []
    count = 0
    filenames = sorted(os.listdir(folder))
    for filename in filenames:
        if num_images is not None and count >= num_images:
            break
        img = Image.open(os.path.join(folder, filename))
        img_array = np.array(img)
        images.append(img_array.flatten())
        count += 1
    return np.array(images), filenames

def main(args):
    image = cv2.imread(args.example_image, cv2.IMREAD_COLOR)
    image_height, image_width, _ = image.shape

    data, filenames = load_images(args.data_folder, args.num_images)
    print("The data is loaded")

    kmedoids = KMedoids(n_clusters=args.n_clusters, metric='euclidean', random_state=0)
    cluster_labels = kmedoids.fit_predict(data)
    print("KMedoids clustering with Euclidean distance completed")

    filename_to_cluster = dict(zip(filenames, cluster_labels))
    for idx, original_filename in enumerate(filename_to_cluster):
        cluster_num = filename_to_cluster[original_filename]
        cluster_folder = os.path.join(args.output_folder, f'cluster_{cluster_num+1}')
        os.makedirs(cluster_folder, exist_ok=True)
        image_path = os.path.join(cluster_folder, original_filename)
        image_reshaped = np.reshape(data[idx], (image_height, image_width, 3)).astype(np.uint8)
        Image.fromarray(image_reshaped).save(image_path)

    print("Clustered images saved.")

    medoid_images = data[kmedoids.medoid_indices_]
    os.makedirs(args.medoid_folder, exist_ok=True)
    for i, medoid_image in enumerate(medoid_images):
        reshaped = medoid_image.reshape((image_height, image_width, 3)).astype(np.uint8)
        filename = os.path.join(args.medoid_folder, f'medoids_cluster_{i+1}.png')
        Image.fromarray(reshaped).save(filename)

    print("Medoid images saved.")

    image_files = [f for f in os.listdir(args.medoid_folder) if f.lower().endswith('.png')]
    os.makedirs(args.threshold_output_folder, exist_ok=True)
    for image_file in image_files:
        image_path = os.path.join(args.medoid_folder, image_file)
        image = cv2.imread(image_path)
        pixel_vals = image.reshape((-1, 3)).astype(np.float32)
        mean_pixel_value = np.mean(pixel_vals)
        _, thresholded_image = cv2.threshold(image, mean_pixel_value, 255, cv2.THRESH_TOZERO_INV)
        mask = (thresholded_image > np.min(thresholded_image))
        thresholded_image[mask] = 255
        thresholded_image_gray = cv2.cvtColor(thresholded_image, cv2.COLOR_BGR2GRAY)
        output_path = os.path.join(args.threshold_output_folder, f'thresholded_{image_file}')
        cv2.imwrite(output_path, thresholded_image_gray)
        plt.subplot(1, 2, 2)
        plt.imshow(thresholded_image_gray, cmap='gray')
        plt.title('Thresholded Image')
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KMedoids Clustering with Euclidean Distance and Thresholding.")
    parser.add_argument("--example_image", type=str, required=True, help="Path to a sample image to get dimensions.")
    parser.add_argument("--data_folder", type=str, required=True, help="Folder containing images to cluster.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save clustered images.")
    parser.add_argument("--medoid_folder", type=str, required=True, help="Folder to save medoid images.")
    parser.add_argument("--threshold_output_folder", type=str, required=True, help="Folder to save thresholded medoid images.")
    parser.add_argument("--num_images", type=int, default=50, help="Number of images to load for clustering.")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters for KMedoids.")
    args = parser.parse_args()
    main(args)
