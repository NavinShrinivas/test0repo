import keras_ocr
import numpy as np
from ultralytics import YOLO
import cv2
import time
import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv, estimate_speed
from model2 import number_plate_detection
import pytesseract

##############################
import Core.utils as utils
from Core.config import cfg
from Core.yolov4 import YOLOv4, decode

from absl import app, flags
from absl.flags import FLAGS
import cv2
import numpy as np
import time

import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0],
#                                                         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])  # limits gpu memory usage

# downloads pretrained weights for text detector and recognizer
pipeline = keras_ocr.pipeline.Pipeline()

tf.keras.backend.clear_session()

STRIDES = np.array(cfg.YOLO.STRIDES)
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, False)
NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
XYSCALE = cfg.YOLO.XYSCALE

##############################


results = {}

mot_tracker = Sort()
prev_bbox_centers = {}

# load model
coco_model = YOLO("yolov8n.pt")
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./sample.mp4')
video_fps = cap.get(cv2.CAP_PROP_FPS)
vehicles = [2, 3, 5, 7]

frame_nmr = -1

# read frames
prev_frame = None
prev_results = None
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

        # detect licence plate
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(
                license_plate, track_ids)

            if car_id != -1:  # Only if the license plate can be recognised it records the vehicle

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]

                # # process license plate
                # license_plate_crop_gray = cv2.cvtColor(
                #     license_plate_crop, cv2.COLOR_BGR2GRAY)
                # _, license_plate_crop_thresh = cv2.threshold(
                #     license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                #
                # # read license plate number
                # license_plate_text, license_plate_text_score = read_license_plate(
                #     license_plate_crop_thresh)
                # num_plate = number_plate_detection(license_plate_crop)
                # gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                # resized = cv2.resize(
                #     gray, (300, 50), interpolation=cv2.INTER_CUBIC)
                # dn_gray = cv2.fastNlMeansDenoising(
                #     resized, templateWindowSize=7, h=25)
                # gray_bin = cv2.threshold(
                #     dn_gray, 80, 255, cv2.THRESH_BINARY)[1]
                # config = ('-l eng --oem 1 --psm 3')
                # num_plate = pytesseract.image_to_string(
                #     gray_bin, config=config)
                # print(num_plate)
                prediction_groups = pipeline.recognize(
                    [license_plate_crop])
                num_plate = ''
                for j in range(len(prediction_groups[0])):
                    num_plate = num_plate + \
                        prediction_groups[0][j][0].upper()[::-1]
                print(num_plate[::-1])

                if num_plate is not None:
                    print("NICEEEE")
                # Get the speed and license plate information for the car
                    car_data = {
                        'locations': track_ids,  # Assuming you have the car's locations over time in track_ids
                    }
                    car_info = estimate_speed(car_id, car_data)

                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'car_speed': car_info['speed_label'],
                        'license_plate': {'bbox': [x1, y1, x2, y2],
                                          'text': num_plate[::-1],
                                          'bbox_score': score,
                                          'text_score': 1}}

# write results
write_csv(results, './speed_test.csv')
