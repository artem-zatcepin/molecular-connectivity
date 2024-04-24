import numpy as np

import nibabel as nib
from radiomics import featureextractor


class Subject:

    def __init__(self,
                 nifti_image_path,
                 ):
        self.nim_path = nifti_image_path
        nim = nib.load(nifti_image_path)
