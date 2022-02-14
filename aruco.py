import os
import cv2
import sys
import time
import pickle
import argparse
import threading
import numpy as np

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

def load_calibration(filename):
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj["width"], obj["height"], obj["camera_matrix"], obj["distortion_coeffs"]

class Visualizer(threading.Thread):
    def __init__(self, owner):
        super(Visualizer, self).__init__()
        self.owner = owner
        self.timestamp = None
        self.current_timestamp = None
        self.image = None
        self.terminated = False
        self.start()

    def run(self):
        while not self.terminated:
            if self.timestamp:
                if self.current_timestamp:
                    if self.timestamp > self.current_timestamp:
                        cv2.imshow("Frame", self.image)
                        self.current_timestamp = self.timestamp
                else:
                    self.current_timestamp = self.timestamp
            if cv2.waitKey(33) == 27:
                self.terminated = True
        self.owner.done = True

class ImageProcessor(threading.Thread):
    def __init__(self, owner, args):
        super(ImageProcessor, self).__init__()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.marker_size = args.marker_size
        self.dictionary = ARUCO_DICT[args.dictionary]
        _, _, self.camera_matrix, self.distortion_coeffs = load_calibration(args.camera_calibration)
        self.reset()
        self.start()

    def run(self):
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    aruco_dict = cv2.aruco.Dictionary_get(self.dictionary)
                    aruco_params = cv2.aruco.DetectorParameters_create()
                    self.corners, self.ids, rejected = cv2.aruco.detectMarkers(self.image, aruco_dict, parameters = aruco_params)
                    if len(self.corners) > 0:
                        self.rvecs, self.tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(self.corners, self.marker_size, self.camera_matrix, self.distortion_coeffs)
                finally:
                    self.event.clear()
                    # Return ourselves to the available pool
                    with self.owner.lock:
                        self.done = True
                        self.owner.pool.append(self)

    def reset(self):
        self.done = False
        self.ids = []
        self.corners = []
        self.rvecs = []
        self.tvecs = []
        self.image = None
        self.timestamp = None
        self.frame = 0

class ProcessOutput(object):
    def __init__(self, args):
        self.done = False
        # Construct a pool of image processors along with a lock
        # to control access between threads
        self.lock = threading.Lock()
        self.pool = [ImageProcessor(self, args) for i in range(args.num_threads)]
        self.processor = None
        self.frame = 0

        self.args = args
        if self.args.visualize:
            self.visualizer = Visualizer(self)

    def print_result(self, processor):
        # assert(processor.done)
        print(processor.timestamp, processor.frame, end=" ")
        if processor.ids is not None:
            for rvec, tvec, id in zip(processor.rvecs, processor.tvecs, processor.ids):
                print(id, rvec, tvec, end=" ")
        print(flush=True)

    def show_result(self, processor):
        # assert(processor.done)
        self.visualizer.timestamp = processor.timestamp
        self.visualizer.image = processor.image

    def store_result(self, processor):
        # assert(processor.done)
        dirName = "aruco"

        if not os.path.isdir(dirName):
            os.mkdir(dirName)

        cv2.imwrite("{}/{}_{}.jpg".format(dirName, processor.timestamp, processor.frame), processor.image)

    def new_frame(self, frame):
        self.frame += 1
        timestamp = str(round(time.time() * 1000))
        with self.lock:
            if self.pool:
                if self.pool[-1].done:
                    if self.args.visualize or self.args.store:
                        if (self.args.visualize or self.args.store) and self.pool[-1].corners:
                            for rvec, tvec, crns, id in zip(self.pool[-1].rvecs, self.pool[-1].tvecs, self.pool[-1].corners, self.pool[-1].ids):
                                cv2.aruco.drawAxis(self.pool[-1].image, self.pool[-1].camera_matrix, self.pool[-1].distortion_coeffs, rvec, tvec, 0.176)
                                cv2.putText(self.pool[-1].image, "id: {} x: {:.2f} y: {:.2f} z: {:.2f}".format(id, tvec[0][0], tvec[0][1], tvec[0][2]),
                                    (int(crns[0][0][0]), int(crns[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
                    if self.args.print_results:
                        self.print_result(self.pool[-1])
                    if self.args.visualize:
                        self.show_result(self.pool[-1])
                    if self.args.store:
                        self.store_result(self.pool[-1])
                    self.pool[-1].reset()
                self.processor = self.pool.pop()
                self.processor.frame = self.frame
                self.processor.image = frame
                self.processor.timestamp = timestamp
                self.processor.event.set()
            else:
                if self.args.print_results:
                    print(timestamp, self.frame, flush=True)
                self.processor = None

    def flush(self):
        # When told to flush (this indicates end of recording), shut
        # down in an orderly fashion. First, add the current processor
        # back to the pool
        print("Terminating ...")
        if self.processor:
            with self.lock:
                self.pool.append(self.processor)
                self.processor = None

        if self.args.visualize:
            self.visualizer.terminated = True
            self.visualizer.join()

        # Now, empty the pool, joining each thread as we go
        while True:
            proc = None
            with self.lock:
                try:
                    proc = self.pool.pop()
                except IndexError:
                    pass # pool is empty
            if not proc:
                break
            proc.terminated = True
            proc.join()

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description='Detects ArUco markers and computes their pose.')
    parser.add_argument('-n', '--num-threads', default=1, type=int)
    parser.add_argument('-p', '--print-results', action='store_const', const=True, default=False)
    parser.add_argument('-v', '--visualize', action='store_const', const=True, default=False)
    parser.add_argument('-s', '--store', action='store_const', const=True, default=False)
    parser.add_argument('-m', '--marker-size', required=True, type=float)
    parser.add_argument('-a', '--dictionary', default="DICT_4X4_50", type=str)
    parser.add_argument('-c', '--camera-calibration', default="camera_calibration.p", type=str)
    parser.add_argument('-d', '--focus-distance', default=0, type=int)
    parser.add_argument('-f', '--fps', default=30, type=int)
    return parser.parse_args(sys.argv[1:])

def focus(val):
    value = (val << 4) & 0x3ff0
    data1 = (value >> 8) & 0x3f
    data2 = value & 0xf0
    os.system("i2cset -y 6 0x0c %d %d" % (data1,data2))

def gstreamer_pipeline (capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0) :
    return ('nvarguscamerasrc ! '
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

if __name__ == "__main__":
    args = parse_command_line_arguments()
    width, height, _, _ = load_calibration(args.camera_calibration)

    cap = cv2.VideoCapture(gstreamer_pipeline(width, height, width, height, args.fps, 0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        focus(args.focus_distance)
        output = ProcessOutput(args)
        while not output.done:
            # capture image
            ret_val, image = cap.read()
            if not ret_val:
                continue
            output.new_frame(image)
        output.flush()
    else:
        print('Unable to open camera')
