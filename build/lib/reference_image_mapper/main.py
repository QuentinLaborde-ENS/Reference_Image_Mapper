# -*- coding: utf-8 -*-

 

import os
import sys
import time

import numpy as np
import pandas as pd
import cv2
import torch 
import func_timeout
import psutil

from matplotlib import pyplot as plt 
import matplotlib.animation as animation
#import matplotlib.animation as animation 

#from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from gluestick.drawing import plot_images, plot_lines, plot_color_line_matches, plot_keypoints, plot_matches
from gluestick.models.two_view_pipeline import TwoViewPipeline 
 
import gluestick
 

 
    

def preprocessing_rim(gaze_data, 
                      time_stamps, 
                      config
                      ):
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
                     outputDir,
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

 
    # Create output directory
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)

    # Copy the reference stim into the output dir
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
                           interpolation=cv2.INTER_LINEAR)
    torch_ref = gluestick.numpy_image_to_torch(ref_frame)
    pred = pipeline_model({'image0': torch_ref.to(device)[None],
                           'image1': torch_ref.to(device)[None]})
    vid = cv2.VideoCapture(world_camera)

    # Keep reference frame descriptors
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
    if warm_start > 0:
        gazeMapped_df = pd.read_csv('{od}/mappedGaze_{n_}.csv'.format(od=outputDir,
                                                                      n_=out_name))
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
    pred = None
    world2ref_transform = None
    current_pred = False

    m_time = config['processing']['max_time']
 
    while vid.isOpened():
        ret, frame = vid.read()
        # memory.append(get_memory_usage())
        if (ret is True) and (frameCounter in framesToCompute) and frameCounter >= warm_start:
            sys.stdout.flush()
            sys.stdout.write("\r Processing frame {i} over {tot}       ".format(i=frameCounter,
                                                                                tot=last_))
         
            
            world_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            world_frame = cv2.resize(world_frame,
                                     down_points,
                                     interpolation=cv2.INTER_LINEAR)
            start_time = time.time()
            try:
                torch_world = gluestick.numpy_image_to_torch(world_frame)
                pred = func_timeout.func_timeout(m_time,
                                                 pipeline_model,
                                                 args=[{'image0': torch_world.to(device)[None],
                                                        'image1': torch_ref.to(device)[None],
                                                        'ref': pred_ref}])
                current_pred = True
                del torch_world
            except:
                print('KILLED: too long')
                current_pred = False
                del torch_world
                pass

            print("--- %s seconds ---" % (time.time() - start_time))
            # If current pred and enough matches, update world2ref_transform
            if current_pred:
                # if current_pred and (pred['matches0'] >= 0).sum() >= config['processing']['min_point_matches'] :
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

                    # For homography
                    valid_matches = line_matches != -1
                    match_indices = line_matches[valid_matches]
                    matched_lines0 = line_seg0[valid_matches]
                    matched_lines1 = line_seg1[match_indices]

                    # Find homography
                    ref2world, mask = cv2.findHomography(matched_kps1,
                                                         matched_kps0,
                                                         cv2.RANSAC,
                                                         10)

                    world2ref_transform = cv2.invert(ref2world)[1]

                except:
                    current_pred = False
                    print('KILLED: could not access prediction')
                    pass

            if current_pred:
                try:
                    number_matches = (pred['matches0'] >= 0).sum()
                    del pred
                except:
                    number_matches = 0
                    current_pred = False
            else:
                number_matches = 0
                current_pred = False
            print('Number matches:')
            print(number_matches)

            # If world2ref_transform already initialized, compute ref gaze
            if world2ref_transform is not None:
                thisFrame_gazeData_world = gazeWorld_df.loc[gazeWorld_df['frame_idx'] == frameCounter]
                world_pts = []
                ref_pts = []

                for i, gazeRow in thisFrame_gazeData_world.iterrows():
                    ts = gazeRow['timestamp']
                    conf = gazeRow['confidence']
                    # Translate normalized gaze data to world pixel coords
                    world_gazeX = gazeRow['norm_pos_x'] * d_w
                    world_gazeY = gazeRow['norm_pos_y'] * d_h
                    world_pts.append([world_gazeX, world_gazeY])

                    ref_gazeX, ref_gazeY = mapCoords2D(
                        (world_gazeX, world_gazeY), world2ref_transform)
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
                    }, index=[i])
                    gazeMapped_df = pd.concat([gazeMapped_df,
                                               thisRow_df])
                to_plot = dict({
                    'image_0': world_pts,
                    'image_1': ref_pts
                })
               

        frameCounter += 1
        if frameCounter > np.max(np.array(framesToCompute)):
            vid.release()

     
    gazeMapped_df.to_csv('{od}/mappedGaze_{n_}.csv'.format(od=outputDir,
                                                           n_=out_name),
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
 
    
 
def display_results(world_camera, 
                    reference_image, 
                    out_name,
                    out_dir):
    
    # Vérifier la version d'OpenCV
    print("OpenCV version:", cv2.__version__)  # Doit afficher 4.7.0
    
    # Chemins
 
    
    output_path = "{od}/video_rim.mp4".format(od = out_dir)
    out_data = '{od}/mappedGaze_{name}.csv'.format(od = out_dir,
                                                   name = out_name)
    
    df_results = pd.read_csv(out_data)
    # Dimensions pour le redimensionnement
    down_width = 600
    down_height = 450
    
    # Vérifier si les fichiers d'entrée existent
    if not os.path.exists(world_camera):
        print(f"Erreur : Le fichier {world_camera} n'existe pas")
        exit()
    if not os.path.exists(reference_image):
        print(f"Erreur : Le fichier {reference_image} n'existe pas")
        exit()
    
    # Créer le dossier de sortie
    os.makedirs("output_test", exist_ok=True)
    
    # Ouvrir la vidéo avec OpenCV
    cap = cv2.VideoCapture(world_camera)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo")
        exit()
    
    # Récupérer les propriétés
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Propriétés vidéo originales : {width}x{height}, FPS: {fps}, Frames: {frame_count}")
    
    # Vérifier les dimensions
    if width == 0 or height == 0:
        print("Erreur : Dimensions non valides")
        cap.release()
        exit()
    
    # Lire l'image de référence
    ref_frame = cv2.imread(reference_image, cv2.IMREAD_COLOR)
    if ref_frame is None:
        print(f"Erreur : Impossible de lire l'image {reference_image}")
        cap.release()
        exit()
    
    # Redimensionner l'image de référence
    ref_frame_resized = cv2.resize(ref_frame, (down_width, down_height), interpolation=cv2.INTER_LINEAR)
    # Convertir l'image de référence en RGB
    ref_frame_rgb = cv2.cvtColor(ref_frame_resized, cv2.COLOR_BGR2RGB)
    
    # Lire les frames de la vidéo
    frames = []
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Fin de la lecture à la frame {frame_number}")
            break
        # Redimensionner la frame
        frame_resized = cv2.resize(frame, (down_width, down_height), interpolation=cv2.INTER_LINEAR)
        # Convertir BGR en RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_number += 1
    
    cap.release()
    
    # Vérifier si des frames ont été lues
    if not frames:
        print("Erreur : Aucune frame lue")
        exit()
    
    # Créer des frames combinées avec points
    frames_combined = []
    for i, frame_rgb in enumerate(frames):
        
        
        local_result_df = df_results[df_results['worldFrame']==i]
     
        ref_gaze_x = local_result_df['ref_gazeX']
        ref_gaze_y = local_result_df['ref_gazeY']
        world_gaze_x = local_result_df['world_gazeX']
        world_gaze_y = local_result_df['world_gazeY']
         
        # Concaténer la frame vidéo (gauche) avec l'image de référence (droite)
        frame_combined = np.hstack((frame_rgb, ref_frame_rgb))
        
        # Calculer les coordonnées du point (basé sur les dimensions redimensionnées)
        y_left = np.mean(world_gaze_y.values)
        y_right = np.mean(ref_gaze_y.values)
        
        
        # Coordonnées pour les deux images
        x_left = np.mean(world_gaze_x.values)
        x_right = np.mean(ref_gaze_x.values) + down_width  # Point sur l'image de référence (droite)
        
        # Stocker la frame et les coordonnées des points
        frames_combined.append((frame_combined, x_left, y_left, x_right, y_right))
    
    # Configurer Matplotlib pour l'animation
    fig, ax = plt.subplots()
    img = ax.imshow(frames_combined[0][0])  # Afficher la première frame combinée
    points_left = ax.scatter([], [], c='red', s=100, marker='o', label='Point gauche')
    points_right = ax.scatter([], [], c='blue', s=100, marker='o', label='Point droit')
    
    ax.set_xticks([])  
    ax.set_yticks([])
    
    # Fonction de mise à jour pour l'animation
    def update(frame_idx):
        frame_data = frames_combined[frame_idx]
        frame_with_points, x_left, y, x_right, y_right = frame_data
        
        # Mettre à jour l'image
        img.set_array(frame_with_points)
        
        # Mettre à jour les points
        points_left.set_offsets([x_left, y])
        points_right.set_offsets([x_right, y_right])
        
        #ax.set_title(f"Frame {frame_idx} (Vidéo gauche, Référence droite)")
        return [img, points_left, points_right]
    
    # Configurer l'animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames_combined),
        interval=1000/fps,  # Intervalle en millisecondes (1000/FPS)
        blit=True
    )
    
    # Configurer l'écrivain FFMpeg pour sauvegarder la vidéo
    writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
    try:
        ani.save(output_path, writer=writer)
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de la vidéo : {e}")
        plt.close()
        exit()
    
    # Fermer Matplotlib
    plt.close()
    print("Animation et écriture terminées")
    
    # Vérifier le fichier de sortie
    if os.path.exists(output_path):
        print(f"Vidéo enregistrée dans {output_path}, taille : {os.path.getsize(output_path)} octets")
    else:
        print("Erreur : Le fichier de sortie n'a pas été créé")
          
 
    
def process_rim(gaze_data, time_stamps, reference_image, 
                world_camera, out_name, out_dir, 
                width_video, height_video,
                display):
 
    config = {
        'processing':{
            'downsampling_factor': 1,
            'max_time': 5,
            'min_line_matches': 0,
            'min_point_matches': 100,
            'camera':{
                'width': width_video,
                'height': height_video,
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
      
    
    preProData = preprocessing_rim(gaze_data, 
                                   time_stamps,
                                   config) 
    processRecording(preProData, 
                     reference_image,
                     world_camera,
                     out_name,
                     out_dir,
                     config) 
    
    if display:
        display_results(world_camera, 
                        reference_image, 
                        out_name,
                        out_dir)
 
    
    
    
    
    
    
    
    
    
    
    
    