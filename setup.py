# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 21:16:42 2024

@author: marca
"""

from setuptools import setup, find_packages
import numpy as np 


 
setup(
    name="reference_image_mapper",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"}, 
    include_dirs=[np.get_include()],
)
