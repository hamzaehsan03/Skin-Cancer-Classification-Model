import numpy as np
import pydicom
import os

current_directory = os.getcwd()
dcm_dir = os.path.join(current_directory, "data\\train\\ISIC_0052212.dcm")


dicom_file = pydicom.read_file(dcm_dir)
print(dicom_file)

import tensorflow as tf

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
