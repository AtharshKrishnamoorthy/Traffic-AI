import streamlit as st
from ultralytics import YOLO
from ultralytics.solutions import speed_estimation, object_counter
import cv2
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import ast
import csv
import easyocr
from moviepy.video.io.VideoFileClip import VideoFileClip
import string
from sort.sort import *
import subprocess
from helper_traffic import *



# Function to process live video for Speed Estimation, Vehicle Counting, or ANPR
def process_video_recorded(selected_task, video_path, start_time, end_time):
    if not os.path.exists(video_path):
            st.error("Video file not found.")
            return

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    
     # Load YOLO model
    model= YOLO("yolov8s.pt")
    names = model.model.names

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    line_pts = [(int(0.2 * w), int(0.5 * h)), (int(0.8 * w), int(0.5 * h))]
    region_points = [(int(0.02 * w), int(0.7 * h)), (int(0.98 * w), int(0.7 * h)),
                     (int(0.98 * w), int(0.9 * h)), (int(0.02 * w), int(0.9 * h))]

    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=True, reg_pts=region_points, classes_names=names, draw_tracks=True)

    speed_obj = speed_estimation.SpeedEstimator()
    speed_obj.set_args(reg_pts=line_pts, names=names, view_img=False)

    output_video_path = "output.mp4"
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            st.write("Video processing has been successfully completed.")
            break

        tracks = model.track(im0, persist=True, show=False)

        if selected_task == 'Speed Estimation':
            im0 = speed_obj.estimate_speed(im0, tracks)
            video_writer.write(im0)
        elif selected_task == 'Vehicle Counting':
            im0 = counter.start_counting(im0, tracks)
            video_writer.write(im0)
        elif selected_task == 'ANPR':

            #Main.py file

            results = {}
            mot_tracker = Sort()

            # load models
            coco_model = YOLO('yolov8n.pt')

            model_path = ('D:\\docs and pdfs\\Traffic Management\\best_w.pt')
            license_plate_detector = YOLO(model_path)


            # load video
            cap = cv2.VideoCapture(video_path)

            vehicles = [2, 3, 5, 7]

            # read frames
            frame_nmr = -1
            ret = True
            while ret:
                frame_nmr += 1
                ret, frame = cap.read()
                if ret:
                   results[frame_nmr] = {}
                   # detect vehicles
                   detections = coco_model(frame)[0]
                   detections_ = []
                   for detection in detections.boxes.data.tolist():
                      x1, y1, x2, y2, score, class_id = detection
                      if int(class_id) in vehicles:
                        detections_.append([x1, y1, x2, y2, score])

                   # track vehicles
                   track_ids = mot_tracker.update(np.asarray(detections_))

                   # detect license plates
                   license_plates = license_plate_detector(frame)[0]
                   for license_plate in license_plates.boxes.data.tolist():
                         x1, y1, x2, y2, score, class_id = license_plate

                         # assign license plate to car
                         xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                         if car_id != -1:
                         # crop license plate
                           license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                         # process license plate
                           license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                           _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
   
                          # read license plate number
                         license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                         if license_plate_text is not None:
                                     results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
                    
            
            

            write_csv(results, 'D:\\docs and pdfs\\Traffic Management\\test.csv')