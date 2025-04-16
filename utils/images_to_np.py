import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
from multiprocessing import Pool, cpu_count

def pil_to_normalized_array(path_label, size=(112, 63)):
    """
    Converts PIL image to a normalized np array
    """
    path, label = path_label
    with Image.open(path) as img:
        img = img.convert('L').resize(size)
        np_img = np.array(img).astype('float32') / 255.0
        return (np_img, label)

def gather_image_paths(folder_path):
    """
    Gathers all image file paths and their labels.
    """
    path_label_list = []
    label_to_idx = {}
    current_label_idx = 0

    for label_name in sorted(os.listdir(folder_path)):
        label_path = os.path.join(folder_path, label_name)
        if not os.path.isdir(label_path):
            continue

        if label_name not in label_to_idx:
            label_to_idx[label_name] = current_label_idx
            current_label_idx += 1

        for filename in os.listdir(label_path):
            file_path = os.path.join(label_path, filename)
            path_label_list.append((file_path, label_to_idx[label_name]))

    return path_label_list, label_to_idx

def process_single_image(args, size=(112, 63)):
    """
    Single image processing to help with parallel computing
    """
    return pil_to_normalized_array(args, size=size)

def process_images_in_parallel(path_label_list, size=(112, 63)):
    """
    Uses multiprocessing to load and process images in parallel.
    """
    cpu_cores = min(cpu_count() - 1, 8)
    print(f"Using {cpu_cores} CPU cores")

    with Pool(cpu_cores) as pool:
        results = list(tqdm(pool.imap_unordered( # tdqm - helpful to see progress
            process_single_image,
            path_label_list
        ), total=len(path_label_list)))

    results = [r for r in results if r is not None]
    images, labels = zip(*results)
    return np.array(images), np.array(labels)

def save_dataset(data, labels, label_map, output_path='image_dataset.npz'):
    """
    Saves the image data and labels into a compressed .npz file, plus label map as pickle.
    """
    np.savez_compressed(output_path, images=data, labels=labels)

    print(label_map.size)
    with open('label_map.pkl', 'wb') as f:
        pickle.dump(label_map, f)

    print(f"Dataset saved to {output_path}")
    print(f"Label map saved to label_map.pkl")

if __name__ == "__main__":
    data_dir = "../out"

    print("Collecting image paths...")
    path_label_list, label_map = gather_image_paths(data_dir)

    print("Processing images in parallel...")
    images, labels = process_images_in_parallel(path_label_list)

    print("Saving dataset...")
    save_dataset(images, labels, label_map)
