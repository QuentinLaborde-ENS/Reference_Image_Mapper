# -*- coding: utf-8 -*-

 
import os
import sys
import time

import numpy as np
import pandas as pd
import cv2
import torch 
import func_timeout

from matplotlib import pyplot as plt
#import matplotlib.animation as animation 

#from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from gluestick.drawing import plot_images, plot_lines, plot_color_line_matches, plot_keypoints, plot_matches
from gluestick.models.two_view_pipeline import TwoViewPipeline 
 
import gluestick
#import homography_est.homography_est as hest
 
    

def preprocessing_rim(gaze_data, 
                      time_stamps, 
                      config):
    '''
    

    Parameters
    ----------
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    output_df : TYPE
        DESCRIPTION.

    '''
 
    
    
    gaze_df = pd.read_csv(gaze_data) 
    gaze_ts = gaze_df['timestamp [ns]'].values/1e6
     
    world_df = pd.read_csv(time_stamps)
    world_ts = world_df['timestamp [ns]'].values/1e6
   
    per_n_frames = config['processing']['downsampling_factor'] 
    frame = (np.ones(len(gaze_ts))*(len(world_ts) - 1)).astype(int)
   
    frame_idx = 0
    gaze_index = 0
    
    while True:
        try: 
            g_ts = gaze_ts[gaze_index]
            w_ts = world_ts[frame_idx] 
            if g_ts <= w_ts: 
                frame[gaze_index] = int(frame_idx)
                gaze_index += 1 
            else:
                frame_idx +=per_n_frames 
        except IndexError:
            break  
        
    gaze_x = gaze_df['gaze x [px]']
    gaze_x /= config['processing']['camera']['width'] 
    gaze_y = gaze_df['gaze y [px]']
    gaze_y /= config['processing']['camera']['height'] 
    
    confidence = gaze_df['worn'] 
    output_df = pd.DataFrame(data=dict({'timestamp': gaze_ts, 
                                        'frame_idx': frame, 
                                        'confidence': confidence, 
                                        'norm_pos_x': gaze_x, 
                                        'norm_pos_y': gaze_y}))
    return output_df

 
 
def processRecording(gazeWorld_df, 
                     reference_image,
                     world_camera,
                     out_name,
                     config, 
                     warm_start=0):
    '''
    

    Parameters
    ----------
    gazeWorld_df : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
     
    frameProcessing_startTime = time.time()
    
    ## Get input data  
    outputDir=config['processing']['files']['outputDir']        
    ## Create output directory
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
 
    ## Copy the reference stim into the output dir 
    framesToCompute = gazeWorld_df['frame_idx'].values.tolist()
    last_ = framesToCompute[-1]
    frameCounter = 0 
    gazeMapped_df = None 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    pipeline_model = TwoViewPipeline(config['model']).to(device).eval()
    line_homography = config['model']['use_lines_homoraphy']
      
    d_w = config['processing']['camera']['down_width']
    d_h = config['processing']['camera']['down_height']
    down_points = (d_w, d_h)
    
    ref_frame = cv2.imread(reference_image, 0) 
    ref_frame = cv2.resize(ref_frame, 
                           down_points, 
                           interpolation= cv2.INTER_LINEAR) 
    torch_ref = gluestick.numpy_image_to_torch(ref_frame)  
    pred = pipeline_model({'image0': torch_ref.to(device)[None], 
                           'image1': torch_ref.to(device)[None]})  
    vid = cv2.VideoCapture(world_camera) 
    
    ## Keep reference frame descriptors 
    pred_ref = dict({})
    for it_ in ['keypoints1', 
                'keypoint_scores1', 
                'descriptors1', 
                'pl_associativity1', 
                'num_junctions1', 
                'lines1', 'orig_lines1', 
                'lines_junc_idx1', 
                'line_scores1',
                'valid_lines1']:
        pred_ref.update({it_[:-1]: pred[it_]})  
    del pred 
    if warm_start>0:
        gazeMapped_df = pd.read_csv('{od}/mappedGaze_{n_}.csv'.format(od = outputDir, 
                                                                      n_ = out_name))
    else:
        gazeMapped_df = pd.DataFrame({
            'gaze_ts': [],
            'worldFrame': [],
            'confidence': [],
            'world_gazeX': [],
            'world_gazeY': [],
            'ref_gazeX': [],
            'ref_gazeY': [], 
            'mapped': [],
            'number_matches': []
            }) 
    print("Processing frames") 
    pred=None
    world2ref_transform=None
    current_pred=False 
   
    m_time = config['processing']['max_time']
    
    while vid.isOpened():   
        ret, frame = vid.read() 
        if (ret is True) and (frameCounter in framesToCompute) and frameCounter >= warm_start:    
            sys.stdout.flush() 
            sys.stdout.write("\r Processing frame {i} over {tot}       ".format(i = frameCounter,
                                                                                tot = last_))
            world_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            world_frame = cv2.resize(world_frame, 
                                     down_points, 
                                     interpolation= cv2.INTER_LINEAR)  
            start_time = time.time()  
            try: 
                torch_world = gluestick.numpy_image_to_torch(world_frame) 
                pred = func_timeout.func_timeout(m_time, 
                                                 pipeline_model,
                                                 args=[{'image0': torch_world.to(device)[None], 
                                                        'image1': torch_ref.to(device)[None],
                                                        'ref': pred_ref}])
                current_pred=True 
                del torch_world 
            except:
                print('KILLED: too long')
                current_pred=False
                del torch_world 
                pass
            
            print("--- %s seconds ---" % (time.time() - start_time))  
            ## If current pred and enough matches, update world2ref_transform
            if current_pred:
            #if current_pred and (pred['matches0'] >= 0).sum() >= config['processing']['min_point_matches'] :
                try:
                    pred = gluestick.batch_to_np(pred)
                    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
                    m0 = pred["matches0"]
                    
                    line_seg0, line_seg1 = pred["lines0"], pred["lines1"]
                    line_matches = pred["line_matches0"]
                    
                    valid_matches = m0 != -1
                    match_indices = m0[valid_matches]
                    matched_kps0 = kp0[valid_matches]
                    matched_kps1 = kp1[match_indices]
                
                    ## For homography
                    valid_matches = line_matches != -1
                    match_indices = line_matches[valid_matches]
                    matched_lines0 = line_seg0[valid_matches]
                    matched_lines1 = line_seg1[match_indices]
    
                    ## Find homography
                    ref2world, mask = cv2.findHomography(matched_kps1, 
                                                         matched_kps0, 
                                                         cv2.RANSAC, 
                                                         10)  
                        
                    world2ref_transform = cv2.invert(ref2world)[1]  
                      
                except:
                    current_pred=False
                    print('KILLED: could not access prediction')
                    pass
                
            if current_pred:
                try:
                    number_matches = (pred['matches0'] >= 0).sum()
                    del pred
                except:
                    number_matches = 0
                    current_pred=False
            else:
                number_matches = 0
                current_pred=False
            print('Number matches:')
            print(number_matches)
            
            ## If world2ref_transform already initialized, compute ref gaze
            if world2ref_transform is not None:
                thisFrame_gazeData_world = gazeWorld_df.loc[gazeWorld_df['frame_idx'] == frameCounter]
                world_pts = []
                ref_pts = []
                
                for i, gazeRow in thisFrame_gazeData_world.iterrows():  
                    ts = gazeRow['timestamp']
                    conf = gazeRow['confidence'] 
                    ## Translate normalized gaze data to world pixel coords 
                    world_gazeX = gazeRow['norm_pos_x'] * d_w
                    world_gazeY = gazeRow['norm_pos_y'] * d_h
                    world_pts.append([world_gazeX, world_gazeY])
                    
                    ref_gazeX, ref_gazeY = mapCoords2D((world_gazeX, world_gazeY), world2ref_transform)
                    ref_pts.append([ref_gazeX, ref_gazeY]) 
                    thisRow_df = pd.DataFrame({
                        'gaze_ts': ts,
                        'worldFrame': frameCounter,
                        'confidence': conf,
                        'world_gazeX': world_gazeX,
                        'world_gazeY': world_gazeY,
                        'ref_gazeX': ref_gazeX,
                        'ref_gazeY': ref_gazeY, 
                        'mapped': current_pred, 
                        'number_matches': number_matches
                        }, index = [i]) 
                    gazeMapped_df = pd.concat([gazeMapped_df, 
                                               thisRow_df]) 
                to_plot = dict({
                    'image_0': world_pts,
                    'image_1': ref_pts
                    }) 
                ## For plotting
                plot_images([cv2.cvtColor(world_frame, cv2.COLOR_GRAY2BGR), 
                             cv2.cvtColor(ref_frame, cv2.COLOR_GRAY2BGR)], 
                            ['World frame {}'.format(frameCounter), 'Reference frame'], 
                            pad=0.5)
                plot_color_line_matches(lw=2, 
                                        points=to_plot)
                plt.show()
                plt.clf()
           
        frameCounter += 1 
        if frameCounter > np.max(np.array(framesToCompute)):  
            vid.release()   
         
        #if frameCounter%10000 == 0: 
        #    gazeMapped_df.to_csv('{od}/mappedGaze_{n_}.csv'.format(od = outputDir, 
        #                                                           n_ = out_name), 
        #                         index=False)
        #    gazeMapped_df = pd.read_csv('{od}/mappedGaze_{n_}.csv'.format(od = outputDir, 
        #                                                                  n_ = out_name))
 
    gazeMapped_df.to_csv('{od}/mappedGaze_{n_}.csv'.format(od = outputDir, 
                                                           n_ = out_name), 
                         index=False) 
     
    endTime = time.time()
    frameProcessing_time = endTime - frameProcessing_startTime
    print('\nTotal time: %s seconds' % frameProcessing_time)
    
        
def mapCoords2D(coords, transform2D): 
    '''
    

    Parameters
    ----------
    coords : TYPE
        DESCRIPTION.
    transform2D : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    '''
   
    ## Reshape input coordinates
    coords = np.array(coords).reshape(-1, 1, 2)  
    ## Compute reference coordinates according to the homography matrix
    mappedCoords = cv2.perspectiveTransform(coords, transform2D)
    mappedCoords = np.round(mappedCoords.ravel(), 3)

    return mappedCoords[0], mappedCoords[1]
 

if __name__ == '__main__':
  
    config = {
        'processing':{
            'downsampling_factor': 20,
            'max_time': 5,
            'min_line_matches': 0,
            'min_point_matches': 100,
            'camera':{
                'width': 1600,
                'height': 1200,
                'down_width': 600,
                'down_height': 450
                },
            'files':{  
                'outputDir': 'mappedGazeOutput',
                }, 
            },
        
        'model' :{
            'name': 'two_view_pipeline',
            'use_lines': True,
            'use_lines_homoraphy': False,
            'extractor': {
                'name': 'wireframe',
                'sp_params': {
                    'force_num_keypoints': False,
                    'max_num_keypoints': 3000,
                },
                'wireframe_params': {
                    'merge_points': True,
                    'merge_line_endpoints': True,
                },
                'max_n_lines': 0,
            },
            'matcher': {
                'name': 'gluestick',
                'weights': str(gluestick.GLUESTICK_ROOT / 'resources' / 'weights' / 'checkpoint_GlueStick_MD.tar'),
                'trainable': False,
            },
            'ground_truth': {
                'from_pose_depth': False,
            }
        }
    }
     
    gaze_data = 'input/_gaze.csv'
    time_stamps = 'input/_world_timestamps.csv'
    reference_image = 'input/_reference_image.jpg'
    world_camera = 'input/_worldCamera.mp4'
    out_name = '_gaze'
    
    preProData = preprocessing_rim(gaze_data, 
                                   time_stamps,
                                   config) 
    processRecording(preProData, 
                     reference_image,
                     world_camera,
                     out_name,
                     config) 
 
    
    
    
    
    
    
    
    
    
    
    
    