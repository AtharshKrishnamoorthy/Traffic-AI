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
import string
from sort.sort import *
import subprocess
from helper_traffic import *
from process_record import *
import time
import matplotlib.pyplot as plt



def process_live_video(selected_task):
    cap = cv2.VideoCapture(0)  # Access the default camera

    # Load YOLO model
    yolo_model = YOLO("yolov8s.pt")
    names = yolo_model.model.names


    # Define the codec and create VideoWriter object to write the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = "output_video.mp4"
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Define region points and line points for vehicle counting and speed estimation
    region_points = [(int(0.02 * w), int(0.7 * h)), (int(0.98 * w), int(0.7 * h)),
                     (int(0.98 * w), int(0.9 * h)), (int(0.02 * +w), int(0.9 * h))]
    line_pts = [(int(0.2 * w), int(0.5 * h)), (int(0.8 * w), int(0.5 * h))]

    # Initialize object counter and speed estimator
    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=False, reg_pts=region_points, classes_names=names, draw_tracks=True)
    speed_estimator = speed_estimation.SpeedEstimator()
    speed_estimator.set_args(reg_pts=line_pts, names=names, view_img=False)


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Perform object detection
        results = yolo_model(frame)

        # Perform speed estimation and vehicle counting
        tracks = yolo_model.track(frame, persist=True, show=False)
        if selected_task == 'Speed Estimation':
             checkbox = st.checkbox(label="Start live processing")
             if checkbox:
                frame_with_speed = speed_estimator.estimate_speed(frame, tracks)
                # Write the processed frame to the output video
                video_writer.write(frame_with_speed)
                cv2.imshow('Processed Frame', frame_with_speed)
        elif selected_task == 'Vehicle Counting':
            checkbox = st.checkbox(label="Start live processing")
            if checkbox:
                frame_with_count = counter.start_counting(frame, tracks)
                # Write the processed frame to the output video
                video_writer.write(frame_with_count)
                cv2.imshow('Processed Frame', frame_with_count)
        elif selected_task == 'ANPR':
            
          checkbox = st.checkbox(label="Start live processing")
          if checkbox:
            # Command to execute the modified script
            command = 'python "D:\\docs and pdfs\\Traffic Management\\run_predictions_with_webcam.py"'

            # Start the modified script using subprocess
            subprocess.run(command, shell=True)

            # Check for 'q' key press to stop the video processing
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        elif selected_task == 'Traffic Rate Calculation':
            model = YOLO("yolov8n.pt")
            cap = cv2.VideoCapture(0)

            frame_count = 0
            start_time = time.time()
            traffic_rates = []

            while cap.isOpened():
                success, frame = cap.read()
            
            if success:
                results = model.track(frame, persist=True)
                annotated_frame = results[0].plot()

                traffic_rate = calculate_traffic_rate(frame_count, start_time, time.time())
                traffic_rates.append(traffic_rate)

                cv2.putText(annotated_frame, f'Traffic Rate: {traffic_rate:.2f} vehicles/hr', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('Frame', annotated_frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                frame_count += 1

            cap.release()
            cv2.destroyAllWindows()

            end_time = time.time()
            traffic_rate = calculate_traffic_rate(frame_count, start_time, end_time)
            traffic_rates.append(traffic_rate)

            st.write("Traffic rate for recorded video: {:.2f} vehicles per hour".format(traffic_rate))

            # Plotting traffic rates
            st.pyplot(plt.plot(range(len(traffic_rates)), traffic_rates))
            plt.xlabel('Frame')
            plt.ylabel('Traffic Rate (vehicles/hr)')
            plt.title('Traffic Rate Over Time')
            plt.show()


    # Release the camera and video writer
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()





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
            
            
            #Add_missing.py 
            
            
            def interpolate_bounding_boxes(data):
            # Extract necessary data columns from input data
                frame_numbers = np.array([int(row['frame_nmr']) for row in data])
                car_ids = np.array([int(float(row['car_id'])) for row in data])
                car_bboxes = np.array([list(map(float, row['car_bbox'].split(','))) for row in data])
                license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'].split(','))) for row in data])

                interpolated_data = []

                unique_car_ids = np.unique(car_ids)
                for car_id in unique_car_ids:
                    frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]
                    print(frame_numbers_, car_id)

                    # Filter data for a specific car ID
                    car_mask = car_ids == car_id
                    car_frame_numbers = frame_numbers[car_mask]
                    car_bboxes_interpolated = []
                    license_plate_bboxes_interpolated = []

                    first_frame_number = car_frame_numbers[0]
                    last_frame_number = car_frame_numbers[-1]

                    for i in range(len(car_bboxes[car_mask])):
                          frame_number = car_frame_numbers[i]
                          car_bbox = car_bboxes[car_mask][i]
                          license_plate_bbox = license_plate_bboxes[car_mask][i]

                          if i > 0:
                             prev_frame_number = car_frame_numbers[i-1]
                             prev_car_bbox = car_bboxes_interpolated[-1]
                             prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                             if frame_number - prev_frame_number > 1:
                                # Interpolate missing frames' bounding boxes
                                frames_gap = frame_number - prev_frame_number
                                x = np.array([prev_frame_number, frame_number])
                                x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                                interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                                interpolated_car_bboxes = interp_func(x_new)
                                interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                                interpolated_license_plate_bboxes = interp_func(x_new)

                                car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                                license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

                          car_bboxes_interpolated.append(car_bbox)
                          license_plate_bboxes_interpolated.append(license_plate_bbox)

                    for i in range(len(car_bboxes_interpolated)):
                        frame_number = first_frame_number + i
                        row = {}
                        row['frame_nmr'] = str(frame_number)
                        row['car_id'] = str(car_id)
                        row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
                        row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

                        if str(frame_number) not in frame_numbers_:
                            # Imputed row, set the following fields to '0'
                            row['license_plate_bbox_score'] = '0'
                            row['license_number'] = '0'
                            row['license_number_score'] = '0'
                        else:
                            # Original row, retrieve values from the input data if available
                            original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
                            row['license_plate_bbox_score'] = original_row['license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
                            row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
                            row['license_number_score'] = original_row['license_number_score'] if 'license_number_score' in original_row else '0'

                        interpolated_data.append(row)

                return interpolated_data
        

            # Load the CSV file
            with open('test.csv', 'r') as file:
               reader = csv.DictReader(file)
               data = list(reader)

            # Interpolate missing data
            interpolated_data = interpolate_bounding_boxes(data)

            # Write updated data to a new CSV file
            header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
            with open('test_interpolated.csv', 'w', newline='') as file:
                 writer = csv.DictWriter(file, fieldnames=header)
                 writer.writeheader()
                 writer.writerows(interpolated_data)
                 
                 
            # Visulaize.py
        

            def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
                x1, y1 = top_left
                x2, y2 = bottom_right

                cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
                cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

                cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
                cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

                cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
                cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

                cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
                cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

                return img


            results = pd.read_csv("D:\\docs and pdfs\\Traffic Management\\test_full.csv")

            # Load video
            cap = cv2.VideoCapture(video_path)

            #  Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

            # Dictionary to store license plate information
            license_plate = {}

            # Extract license plate information for each car
            for car_id in np.unique(results['car_id']):
                 max_score = np.amax(results[results['car_id'] == car_id]['license_number_score'])
                 license_plate[car_id] = {'license_crop': None,
                                         'license_plate_number': results[(results['car_id'] == car_id) &
                                         (results['license_number_score'] == max_score)]['license_number'].iloc[0]}
                 cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                             (results['license_number_score'] == max_score)]['frame_nmr'].iloc[0])
                 ret, frame = cap.read()

                 x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                              (results['license_number_score'] == max_score)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
    
                 # Crop license plate
                 license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                 license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

                 license_plate[car_id]['license_crop'] = license_crop

            frame_nmr = -1

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Read frames
            ret = True
            while ret:
              ret, frame = cap.read()
              frame_nmr += 1
              if ret:
                 df_ = results[results['frame_nmr'] == frame_nmr]
                 for row_indx in range(len(df_)):
                     car_bbox = df_.iloc[row_indx]['car_bbox']
                     license_plate_bbox = df_.iloc[row_indx]['license_plate_bbox']
                     license_plate_bbox_score = df_.iloc[row_indx]['license_plate_bbox_score']
                     license_number = df_.iloc[row_indx]['license_number']
                     license_number_score = df_.iloc[row_indx]['license_number_score']
            
                     if car_bbox != 0 and license_plate_bbox != 0 and license_plate_bbox_score != 0 and license_number != 0 and license_number_score!= 0:
                     # Continue processing the frame only if both car_bbox and license_plate_bbox are not equal to 0
                     # Draw car bounding box
                      car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                      draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                            line_length_x=200, line_length_y=200)

                     # Draw license plate bounding box
                      x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                      cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

                     # Crop license plate
                      if license_plate[df_.iloc[row_indx]['car_id']]['license_crop'] is not None:
                            license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

                            H, W, _ = license_crop.shape

                            try:
                                frame[int(car_y1) - H - 100:int(car_y1) - 100,
                                int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                                frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                                int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                                (text_width, text_height), _ = cv2.getTextSize(
                                   license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    4.3,
                                    17)

                                cv2.putText(frame,
                                     license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                                     (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                                     cv2.FONT_HERSHEY_SIMPLEX,
                                     4.3,
                                     (0, 0, 0),
                                     17)

                            except Exception as e:
                               print(f"Error processing frame {frame_nmr}: {e}")



                 out.write(frame)
                 frame = cv2.resize(frame, (1280, 720))

            out.release()
            cap.release()
            
        elif selected_task == 'Traffic Rate Calculation':
            model = YOLO("yolov8n.pt")
            cap = cv2.VideoCapture(video_path)

            frame_count = 0
            start_time = time.time()
            traffic_rates = []

            while cap.isOpened():
              success, frame = cap.read()
              
              if success:
                results = model.track(frame, persist=True)
                annotated_frame = results[0].plot()

                traffic_rate = calculate_traffic_rate(frame_count, start_time, time.time())
                traffic_rates.append(traffic_rate)

                cv2.putText(annotated_frame, f'Traffic Rate: {traffic_rate:.2f} vehicles/hr', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('Frame', annotated_frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                   break

                frame_count += 1

            cap.release()
            cv2.destroyAllWindows()

            end_time = time.time()
            traffic_rate = calculate_traffic_rate(frame_count, start_time, end_time)
            traffic_rates.append(traffic_rate)

            st.write("Traffic rate for recorded video: {:.2f} vehicles per hour".format(traffic_rate))

             # Plotting traffic rates
            fig, ax = plt.subplots()
            ax.plot(range(len(traffic_rates)), traffic_rates)
            ax.set_xlabel('Frame')
            ax.set_ylabel('Traffic Rate (vehicles/hr)')
            ax.set_title('Traffic Rate Over Time')
            st.pyplot(fig)


    # Release the camera and video writer
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()