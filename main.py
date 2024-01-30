import numpy as np
import pydicom
import os

current_directory = os.getcwd()
dcm_dir = os.path.join(current_directory, "data")


dicom_file = pydicom.read_file(dcm_dir, "train\\ISIC_0052212.dcm")
print(dicom_file)