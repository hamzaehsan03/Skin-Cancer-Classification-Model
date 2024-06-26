import os
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import pandas as pd
from multiprocess import parallel_process


csv_path = ".\\GroundTruth.csv"
metadata = pd.read_csv(csv_path)

folders = ['1', '0']
for folder in folders:
    os.makedirs(f'data/train/{folder}', exist_ok=True)
    os.makedirs(f'data/test/{folder}', exist_ok=True)


# Adjust image size to be consistent with expected input of pre-trained models
def read_dcm(path, img_size = (224, 224)):
    dicom = pydicom.read_file(path)
    data = apply_voi_lut(dicom.pixel_array, dicom)

    if len(data.shape) == 2:
        data = np.stack((data) * 3, axis = -1)

    # Shift the pixel values of the image by subtracting the minimum pixel value in the image
    # This will cause the lowest pixel value to become 0
    data = data - np.min(data)

    # Scale the pixel values to a range from 0 to 1 by dividing it by the maximum pixel value
    # Essentially normalising the image intensity
    data = data / np.max(data)
    # Scale the image to an 8-bit image
    data = (data * 255).astype(np.uint8)
    resized_image = tf.image.resize(data, img_size)
    resized_image = resized_image.numpy().astype(np.uint8)
    # resized_image = tf.cast(resized_image, tf.uint8)
    return resized_image
    
def process_image(image_path, metadata, base_dir, visualise=False):
    image = read_dcm(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    row = metadata[metadata['image_name'] == base_name]
    label = row['target'].values[0] if not row.empty else -1

    label_dir = os.path.join(base_dir, str(label))
    os.makedirs(label_dir, exist_ok=True)

    save_image = os.path.join(label_dir, base_name + '.png')
    plt.imsave(save_image, image, cmap='gray')

    if visualise:
        plt.imshow(image)
        plt.title(f'Label: {label}')
        plt.show()

    return image, label

def main():

    sample_paths = [os.path.join(".\\data\\train", filename) for filename in os.listdir(".\\data\\train")
                    if filename.endswith(".dcm")][:5]
    for path in sample_paths:
        process_image(path, metadata, "data/train", visualise=True)

    def parallel_load_images(folder, metadata, base_dir):
        image_paths = [os.path.join(folder, filename) for filename in os.listdir(folder) if filename.endswith(".dcm")]
        args = [(path, metadata, base_dir) for path in image_paths]

        results = parallel_process(process_image, args, processors=12)
        images, labels = zip(*results)
        return list(images), list(labels)

    current_directory = os.getcwd()
    dcm_dir = os.path.join(current_directory, "data")
    train_images = parallel_load_images(".\\data\\train", metadata, "data/train")
    test_images = parallel_load_images(".\\data\\test", metadata, "data/test")

if __name__ == "__main__":
    main()
