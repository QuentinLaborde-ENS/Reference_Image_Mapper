#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 20:12:57 2025

@author: quentinlaborde
"""
import pandas as pd 
import cv2
import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import reference_image_mapper as rim

rim.process_rim(gaze_data = 'test/input_rim/_gaze.csv',
                    time_stamps = 'test/input_rim/_world_timestamps.csv',
                    reference_image = 'test/input_rim/_reference_image.jpg',
                    world_camera = 'test/input_rim/_worldCamera.mp4',
                    out_name = '_gaze',
                    out_dir = 'test/output_test', 
                    width_video=1600, 
                    height_video=1200,
                    display=True)
                   
           
                   
           

 