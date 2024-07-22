import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import multiprocessing
import numpy as np
import setproctitle
import cv2
import time
import hailo
from yolo_deepsort_with_roi import *
from hailo_rpi_common import (
    get_default_parser,
    QUEUE,
    get_caps_from_pad,
    get_numpy_from_buffer,
    GStreamerApp,
    app_callback_class,
)

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    start = time.time()
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK
        
    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"
    
    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame : np.ndarray = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    det_xyxy_list = []
    
    # Parse the detections
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()

        det_xyxy = (int(bbox.xmin() * width), 
                    int(bbox.ymin() * height), 
                    int(bbox.xmax() * width), 
                    int(bbox.ymax() * height))

        det_xywh = (int(bbox.xmin() * width), 
                    int(bbox.ymin() * height), 
                    int(bbox.width() * width), 
                    int(bbox.height() * height))

        det_xyxy_list.append([det_xywh, confidence, label])

        if label == "person":
            string_to_print += (f"Detection: {label} {confidence:.2f}\n")
            # Pose estimation landmarks from detection (if available)
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if len(landmarks) != 0:
                points = landmarks[0].get_points()
                left_hip = points[11]
                right_hip = points[12]
                left_shoulder = points[5]
                right_shoulder = points[6]
                # The landmarks are normalized to the bounding box, we also need to convert them to the frame size
                left_hip_x = int((left_hip.x() * bbox.width() + bbox.xmin()) * width)
                left_hip_y = int((left_hip.y() * bbox.height() + bbox.ymin()) * height)
                right_hip_x = int((right_hip.x() * bbox.width() + bbox.xmin()) * width)
                right_hip_y = int((right_hip.y() * bbox.height() + bbox.ymin()) * height)
                left_shoulder_x = int((left_shoulder.x() * bbox.width() + bbox.xmin()) * width)
                left_shoulder_y = int((left_shoulder.y() * bbox.height() + bbox.ymin()) * height)
                right_shoulder_x = int((right_shoulder.x() * bbox.width() + bbox.xmin()) * width)
                right_shoulder_y = int((right_shoulder.y() * bbox.height() + bbox.ymin()) * height)
                #string_to_print += (f" Left hip: x: {left_hip_x:.2f} y: {left_hip_y:.2f} Right hip: x: {right_hip_x:.2f} y: {right_hip_y:.2f}\n")
                if user_data.use_frame:    
                    # Add markers to the frame to show hip landmarks
                    display_rectangle(frame, det_xyxy)
                    cv2.circle(frame, (left_hip_x, left_hip_y), 5, (0, 255, 0), -1)
                    cv2.circle(frame, (right_hip_x, right_hip_y), 5, (0, 255, 0), -1)
                    cv2.circle(frame, (left_shoulder_x, left_shoulder_y), 5, (0, 255, 0), -1)
                    cv2.circle(frame, (right_shoulder_x, right_shoulder_y), 5, (0, 255, 0), -1)
                    # Note: using imshow will not work here, as the callback function is not running in the main thread
            
    if user_data.use_frame:
        # Convert the frame to BGR
        # t1 = time.time()
        # tracks : list[Track] = tracker.update_tracks(det_xyxy_list, frame=frame)
        # t2 = time.time()
        # roi = process_tracks(frame, tracks, tracker) 
        end = time.time()
        # print(t2-t1, end - t2)
        fps = 1/(end-start)
        cv2.putText(frame, f"{fps:.2f} fps", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK
    


# This function can be used to get the COCO keypoints coorespondence map
def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    keypoints = {
        'nose': 1,
        'left_eye': 2,
        'right_eye': 3,
        'left_ear': 4,
        'right_ear': 5,
        'left_shoulder': 6,
        'right_shoulder': 7,
        'left_elbow': 8,
        'right_elbow': 9,
        'left_wrist': 10,
        'right_wrist': 11,
        'left_hip': 12,
        'right_hip': 13,
        'left_knee': 14,
        'right_knee': 15,
        'left_ankle': 16,
        'right_ankle': 17,
    }

    return keypoints
#-----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# This class inherits from the hailo_rpi_common.GStreamerApp class

class GStreamerPoseEstimationApp(GStreamerApp):
    def __init__(self, args, user_data):
        # Call the parent class constructor
        super().__init__(args, user_data)
        # Additional initialization code can be added here
        # Set Hailo parameters these parameters should be set based on the model used
        self.batch_size = 2
        self.network_width = 640
        self.network_height = 640
        self.network_format = "RGB"
        # self.default_postprocess_so = os.path.join(self.postprocess_dir, 'libyolov8pose_post.so')
        self.post_function_name = "filter"
        # self.yolo_hef_path = os.path.join(self.current_path, '../resources/yolov8s_pose_h8l_pi.hef')
        self.yolo_hef_path = os.path.join(self.current_path, '../resources/yolov8s_h8l.hef')
        # self.re_id_hef_path = os.path.join(self.current_path, '../resources/repvgg_a0_person_reid_512.hef')
        self.re_id_hef_path = os.path.join(self.current_path, '../resources/osnet_x1_0.hef')
        self.app_callback = app_callback
        self.cropper_so = os.path.join(self.current_path, "../resources/cropper/libre_id.so")
        self.re_id_post_so = os.path.join(self.current_path, "../resources/libre_id.so")
        self.re_id_dewarp_so = os.path.join(self.current_path, "../resources/libre_id_dewarp.so") # temporaire (voir pour modifier le setup.sh)
        self.re_id_overlay_so = os.path.join(self.current_path, "../resources/libre_id_overlay.so")

        new_postprocess_path = os.path.join(self.current_path, '../resources/libyolo_hailortpp_post.so')
        if os.path.exists(new_postprocess_path):
            self.default_postprocess_so = new_postprocess_path
        else:
            self.default_postprocess_so = os.path.join(self.postprocess_dir, 'libyolo_hailortpp_post.so')
        
        # Set the process title
        setproctitle.setproctitle("Hailo Pose Estimation App")

        self.create_pipeline()

    def yolo_pipeline(self):
        pipeline = "tee name=t ! "
        pipeline += QUEUE("bypass_queue", max_size_buffers=20, leaky="no") + "hmux.sink_0 "
        pipeline += "t. ! " + QUEUE("queue_hailonet")
        pipeline += QUEUE("queue_yolo_scale")
        pipeline += f"videoscale ! "
        pipeline += f"video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height} ! "
        pipeline += "videoconvert n-threads=3 ! "
        pipeline += f"hailonet hef-path={self.yolo_hef_path} batch-size={self.batch_size} force-writable=true vdevice-key=1 ! "
        pipeline += QUEUE("queue_hailofilter")
        pipeline += f"hailofilter function-name={self.post_function_name} so-path={self.default_postprocess_so} qos=false ! "
        pipeline += QUEUE("queue_hmux") + " hmux.sink_1 hmux. ! "
        return pipeline
    
    def reid_pipeline(self):
        # pipeline = "tee name=cropper2 ! "
        pipeline = f"hailocropper so-path={self.cropper_so} function-name=create_crops internal-offset=true name=cropper2 "
        pipeline += "hailoaggregator name=hmux2 "
        pipeline += "cropper2. ! " + QUEUE("bypass_queue_reid", max_size_buffers=20, leaky="no") + "hmux2.sink_0 "
        pipeline += "cropper2. ! queue name=pre_reid_q leaky=no max-size-buffers=10 max-size-bytes=0 max-size-time=0 ! "
        pipeline += QUEUE("queue_src_scale2", leaky="no")
        pipeline += f"videoscale ! "
        pipeline += f"video/x-raw, format={self.network_format}, width=128, height=256 ! "
        pipeline += "videoconvert n-threads=3 ! "
        pipeline += f"hailonet hef-path={self.re_id_hef_path} force-writable=true vdevice-key=1 ! "
        pipeline += "queue name=reid_post_q leaky=no max-size-buffers=10 max-size-bytes=0 max-size-time=0 ! "
        pipeline += f"hailofilter so-path={self.re_id_post_so} qos=false ! "
        pipeline += QUEUE("queue_hmux2", leaky="no") + " hmux2.sink_1 hmux2. ! "
        return pipeline

    def get_pipeline_string(self):
        if (self.source_type == "rpi"):
            source_element = f"libcamerasrc name=src_0 auto-focus-mode=2 ! "
            source_element += f"video/x-raw, format={self.network_format}, width=2304, height=1296 ! "
            source_element += QUEUE("queue_src_scale")
            source_element += f"videoscale ! "
            source_element += f"video/x-raw, format={self.network_format}, width=576, height=324 ! "
            # source_element += f"rotate angle=3.1415 ! "
        
        elif (self.source_type == "usb"):
            source_element = f"v4l2src device={self.video_source} name=src_0 ! "
            source_element += f"video/x-raw, width=640, height=480, framerate=30/1 ! "
        else:  
            source_element = f"filesrc location={self.video_source} name=src_0 ! "
            source_element += QUEUE("queue_dec264")
            source_element += f" qtdemux ! h264parse ! avdec_h264 max-threads=2 ! "
            source_element += f" video/x-raw,format=I420 ! "        
        
        pipeline_string = "hailomuxer name=hmux "
        pipeline_string += source_element
        pipeline_string += self.yolo_pipeline()

        pipeline_string += "queue name=hailo_pre_tracker leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \
        hailotracker name=hailo_tracker hailo-objects-blacklist=hailo_landmarks,hailo_depth_mask,hailo_class_mask,hailo_matrix \
        class-id=1 kalman-dist-thr=0.7 iou-thr=0.8 init-iou-thr=0.3 keep-new-frames=2 keep-tracked-frames=1000 \
        keep-lost-frames=8 qos=false ! "

        pipeline_string += self.reid_pipeline()
        pipeline_string += "queue name=hailo_pre_gallery leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \
        hailogallery similarity-thr=.4 gallery-queue-size=100 class-id=1 ! \
        queue name=hailo_post_gallery leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "

        pipeline_string += f"queue name=hailo_pre_draw leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \
        hailofilter use-gst-buffer=true so-path={self.re_id_overlay_so} qos=false ! \
        queue name=hailo_post_draw leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "


        pipeline_string += QUEUE("queue_hailo_python")

        pipeline_string += QUEUE("queue_user_callback")
        pipeline_string += f"identity name=identity_callback ! "
        pipeline_string += QUEUE("queue_hailooverlay")
        pipeline_string += f"hailooverlay ! "
        pipeline_string += QUEUE("queue_videoconvert")
        pipeline_string += f"videoconvert n-threads=3 qos=false ! "
        pipeline_string += QUEUE("queue_hailo_display")
        pipeline_string += f"fpsdisplaysink video-sink={self.video_sink} name=hailo_display sync={self.sync} text-overlay={self.options_menu.show_fps} signal-fps-measurements=true "
        # pipeline_string += f"fpsdisplaysink video-sink=fakesink name=fakesink sync={self.sync} text-overlay={self.options_menu.show_fps} signal-fps-measurements=true "

        print(pipeline_string)
        return pipeline_string
    
if __name__ == "__main__":
    # Create an instance of the user app callback class
    
    user_data = user_app_callback_class()
    parser = get_default_parser()
    args = parser.parse_args()

    i = 0

    max_age = 30
    n_init = 10
    is_stream = True
    init_dist = 2

    frames_to_skip = 1

    # Initialize the DeepSORT object tracker
    tracker = DeepSort(max_age=max_age, n_init=n_init, max_cosine_distance=0.25, half=True)

    # Initialize a list of all the desired keypoints for display
    desired_kpts = [5, 6, 11, 12]

    app = GStreamerPoseEstimationApp(args, user_data)
    app.run()
