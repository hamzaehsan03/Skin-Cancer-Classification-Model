import os
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import tensorflow as tf 
import matplotlib as plt


# Adjust image size to be consistent with expected input of pre-trained models
def read_dcm(path, img_size = (224, 224)):
    dicom = pydicom.read_file(path)
    data = apply_voi_lut(dicom.pixel_array, dicom)

    # If the dicom data is stored in a photometric interpretation of MONOCHROME1
    # Invert the image through subtracting the maximum pixel value from each pixel
    if dicom.PhotometricInterpretation == 'MONOCHROME1':
        data = np.amax(data) - data

    # Shift the pixel values of the image by subtracting the minimum pixel value in the image
    # This will cause the lowest pixel value to become 0
    data = data - np.min(data)

    # Scale the pixel values to a range from 0 to 1 by dividing it by the maximum pixel value
    # Essentially normalising the image intensity
    data = data / np.max(data)

    # Scale the image to an 8-bit image
    data = (data * 255).astype(np.uint8)
    return tf.image.resize(data, img_size)
    
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".dcm"):
            image_path = os.path.join(folder, filename)
            image = read_dcm(image_path)
            images.append(image)
    return images

current_directory = os.getcwd()
dcm_dir = os.path.join(current_directory, "data")
train_images = load_images(dcm_dir, "train")
test_images = load_images(dcm_dir, "test")