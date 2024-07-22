#!/usr/bin/venv python3

"""
SORT : https://arxiv.org/abs/1602.00763
DeepSORT : https://arxiv.org/abs/1703.07402


"""

import time
import cv2
import os
import sys
import torch
import argparse
import gsthailo
from picamera2 import Picamera2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.track import Track
from math import dist as dist

DETECTION_COLOR = (0, 0, 255)
PREDICTION_COLOR = (255, 0, 0)
VALIDATED_COLOR = (0, 255, 0)
ROI_COLOR = (255, 0, 255)

def measure_kpt_dist(keypoints : dict[str, tuple]) -> dict[str, float]:
    """
    Parameters
    ----------
    keypoints : Dict of the 2D keypoint coordinates with its associated name

    Returns
    -------
    Dict of all the chest quadrilateral edges measurements with their associated position
    """
    measures = {
        "left" : dist(keypoints["l_shoulder"][0], keypoints["l_hip"][0]),
        "right" : dist(keypoints["r_shoulder"][0], keypoints["r_hip"][0]),
        "top" : dist(keypoints["l_shoulder"][0], keypoints["r_shoulder"][0]),
        "down" : dist(keypoints["l_hip"][0], keypoints["r_hip"][0])
    }

    return measures

def distance(keypoints : dict[str, tuple], init_measures : dict[str, float], init_distance: float)  -> float:
    d = (init_measures["right"] + init_measures["left"])/2
    new_d = (keypoints["right"] + keypoints["left"])/2
    
    return init_distance * d / new_d


def crop_frame(frame : cv2.typing.MatLike, roi : tuple[int, int, int, int]) -> cv2.typing.MatLike:
    """
    Parameters
    ---------- 
    frame : current frame to be processed
    roi : tuple with the top-left and bottom-right corner of the selected region

    Returns
    -------
    Matrix of the portion of the whole frame matrix
    """
    if roi:
        x1, y1, x2, y2 = roi
        cropped_frame = frame[y1:y2, x1:x2]
        if cropped_frame.size == 0:
            cropped_frame = frame
            roi = None
        print(cropped_frame.shape[0:2])
    else:
        cropped_frame = frame
    return cropped_frame

def display_rectangle(frame : cv2.typing.MatLike, rectangle : tuple[int, int, int, int], label : str = None, color : tuple[int, int, int] = (0, 0, 255)):
    """
    Parameters
    ----------
    frame : current frame to be processed
    rectangle : tuple with the top-left and bottom-right corner of the rectangle
    label : string to be displayed on top of the rectangle
    color : color of the rectangle and the label

    """
    if rectangle is None:
        return 
    x1, y1, x2, y2 = rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if label is not None:
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def process_keypoints(keypoints : torch.Tensor, desired_kpts : list[int], roi) -> dict[str, list[tuple[int], float]]:
    """
    Parameters
    ----------
    keypoints: list of keypoints coordinates and conf value
    desired_kpts : list of the index of the desired keypoints

    Returns
    -------
    Dict of the desired keypoints with the coordinates and the conf value with the associated name
    """
    # Initialize the keypoints dictionnary
    kpts = {}
    for i_kpt, keypoint in enumerate(keypoints):
        if (keypoint[0].item() != 0 and keypoint[1] != 0 and i_kpt in desired_kpts):
            if roi:
                # Adjust coordinates relative to the full frame
                kx, ky = int(keypoint[0].item()) + roi[0], int(keypoint[1].item()) + roi[1]
            else:
                kx, ky = int(keypoint[0].item()), int(keypoint[1].item())

            key_name: str = {5: "l_shoulder", 
                                6: "r_shoulder", 
                                11: "l_hip", 
                                12: "r_hip"}.get(i_kpt)
            
            kpts[key_name] = [(kx, ky), keypoint[2].item()]

    return kpts

def display_keypoints(frame : cv2.typing.MatLike, kpts : dict[str, list[tuple[int], float]] = None):
    """
    Parameters
    ----------
    frame : current frame to be processed
    kpts : dict of the keypoints with the coordinates and the conf value with the associated name
    """
    if kpts is not None:
        for kpt in kpts.values():
            cv2.circle(frame, kpt[0], radius=0, color=DETECTION_COLOR, thickness=8)

def process_results(frame : cv2.typing.MatLike, results : list[YOLO], roi, desired_kpts) -> tuple[list[list[int], float, int], dict[str, list[tuple[int]]] | None]:
    """
    Parameters
    ----------
    frame : current frame to be processed
    results : list of all the objects detected with their corresponding bounding
    boxes and keypoints

    Returns
    -------
    list of the coordinates, conf value and COCO class index and the dict of all the keypoints coordinates
    """
    # Initialize the detections list
    detections = []
    kpts = None

    for result in results:
        # Extract and display the detected bounding box
        boxes = result.boxes
        person_keypoints = result.keypoints
        for box, keypoints in zip(boxes, person_keypoints.data):
            cx1, cy1, cx2, cy2 = map(int, box.xyxy[0])
            if roi :
                x1, y1, x2, y2 = cx1 + roi[0], cy1 + roi[1], cx2 + roi[0], cy2 + roi[1]
            else:
                x1, y1, x2, y2 = cx1, cy1, cx2, cy2
            display_rectangle(frame, (x1, y1, x2, y2))
            cls = int(box.cls)
            conf : float = box.conf
            if cls == 0 and conf > 0.5:
                detections.append([[x1, y1, x2 - x1, y2 - y1], conf, cls])
            
            kpts = process_keypoints(keypoints, desired_kpts, roi)
            display_keypoints(frame, kpts)

    return detections, kpts


def process_tracks(frame : cv2.typing.MatLike, tracks : list[Track], tracker) -> tuple[int, int, int, int]:
    """
    Parameters
    ----------
    frame : current frame to be processed
    tracks : list of all the tracked targets

    Returns
    -------
    tuple with the top-left and bottom-right corner of the region of interest
    """
    for track in tracks:
        # check if the target has been detected on the current frame
        tx1, ty1, tx2, ty2 = map(int, track.to_ltwh())
        tx2, ty2 = tx1 + tx2, ty1 + ty2
        label = f"Person {track.track_id}"
        if track.track_id == '1':
            # print(len(track.features)) 
            if not track.is_confirmed() or track.time_since_update > 1:
                display_rectangle(frame, (tx1, ty1, tx2, ty2), label, PREDICTION_COLOR)
            else:
                display_rectangle(frame, (tx1, ty1, tx2, ty2), label, VALIDATED_COLOR)

            height, width, channels = frame.shape

            if tx1 > tx2 or track.time_since_update >= track._max_age:
                roi = (0, 0, width, height)
                tracks.pop(0)
                tracker.delete_all_tracks()
            else:
                roi = (max(int(tx1 - tx2 * 0.4), 0), 
                    max(int(ty1 - ty2 * 0.2), 0), 
                    min(int(tx2 + tx2 * 0.4), width), 
                    min(int(ty2 + ty2 * 0.2), height))
            return roi


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='TrackPi',
        description='Launch the rpi5 tracker'
    )

    # parser.add_argument('TrackPi')
    parser.add_argument('-i', '--n_init',type=int, default=10)
    parser.add_argument('-m', '--max_age',type=int, default=30)
    parser.add_argument('-s', '--stream', action="store_false")
    parser.add_argument('-d', '--init_dist',type=float, default=2.0)
    parser.add_argument('-f', '--focal_dist', type=float, default=4.74)
    args = parser.parse_args()

    if not args.stream:
        print("video")

    max_age = args.max_age
    n_init = args.n_init
    is_stream = args.stream
    init_dist = args.init_dist

    # Load the exported NCNN model
    model_path = f"{os.path.dirname(__file__)}/yolov8n-pose_ncnn_model"
    model = YOLO(model_path, task="pose")

    # Open the video file
    if not is_stream:
        video_path = f'{os.path.dirname(__file__)}/video_test.mp4'
        cap = cv2.VideoCapture(video_path)

        orig_fps  = cap.get(cv2.CAP_PROP_FPS)

        vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))/orig_fps

        print(orig_fps)
        minutes = int(vid_len/60)
        seconds = vid_len%60

    #   print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

    frames_to_skip = 1

    # Initialize the DeepSORT object tracker
    tracker = DeepSort(max_age=max_age, n_init=n_init, max_cosine_distance=0.25, half=True)

    # Initialize a list of all the desired keypoints for display
    desired_kpts = [5, 6, 11, 12]
    # desired_kpts = range(17)

    roi : tuple[int, int, int, int] = None

    i = 0

    with Picamera2() as picam2:
        if is_stream:
            picam2.start()
        
        # Process the video frames
        while True:
            # Lower the frame rate by passing images in the video
            if is_stream:
                frame = picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                i += 1
            else:
                tracker.tracker.max_age = max_age//frames_to_skip
                tracker.tracker.n_init = n_init//frames_to_skip

                if i % frames_to_skip != 0:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    i += 1
                    continue
                ret, frame = cap.read()
                if not ret:
                    break
                i += 1

            # Initialize the counter to evaluate the number of frames per second 
            start = time.time()

            cropped_frame: cv2.typing.MatLike = crop_frame(frame, roi)

            display_rectangle(frame, roi, "ROI", ROI_COLOR)

            # Apply the YOLO model on the current frame
            results : list[YOLO] = model.predict(cropped_frame, cropped_frame.shape[0:2], device='cpu')
            detections = process_results(frame, results)

            if i <= n_init and detections[1] is not None:
                init_measures = measure_kpt_dist(detections[1])
            elif detections[1] is not None and len(detections[1]) == 4:
                print(detections[1])
                current_measures = measure_kpt_dist(detections[1])
                d = distance(current_measures, init_measures, init_dist)
                print(d)


            # Updates the tracker with new detections
            # If certains target are not matching one of the new detections
            # (in case of occlusion) they're updated with the prediction
            # (only for a limited number of frames)
            tracks : list[Track] = tracker.update_tracks(detections[0], frame=frame)
            roi = process_tracks(frame, tracks, tracker)
            
            end = time.time()
            processing_time = (end - start)
            fps = 1/processing_time

            # Permits to skip frame by frame on debug mode 
            if i != 1 and not getattr(sys, 'gettrace', None)():
                frames_to_skip = int(processing_time * 30)
            else:
                frames_to_skip = 1

            cv2.putText(frame, f"{fps:.2f} fps", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)

            # print(i)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    cv2.destroyAllWindows()