import io
import os
import cv2
import sys
import time
import pickle
import argparse
import picamera
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

def LoadCalibration(filename):
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
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.marker_size = args.marker_size
        self.dictionary = ARUCO_DICT[args.dictionary]
        _, _, self.camera_matrix, self.distortion_coeffs = LoadCalibration(args.camera_calibration)
        self.Reset()
        self.start()

    def run(self):
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    self.stream.seek(0)
                    self.image = cv2.imdecode(np.frombuffer(self.stream.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
                    aruco_dict = cv2.aruco.Dictionary_get(self.dictionary)
                    aruco_params = cv2.aruco.DetectorParameters_create()
                    self.corners, self.ids, rejected = cv2.aruco.detectMarkers(self.image, aruco_dict, parameters = aruco_params)
                    if len(self.corners) > 0:
                        self.rvecs, self.tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(self.corners, self.marker_size, self.camera_matrix, self.distortion_coeffs)
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the available pool
                    with self.owner.lock:
                        self.done = True
                        self.owner.pool.append(self)
    
    def Reset(self):
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

    def PrintResult(self, processor):
        # assert(processor.done)
        print(processor.timestamp, processor.frame, end=" ")
        for rvec, tvec, _ in zip(processor.rvecs, processor.tvecs, processor.corners):
            print(rvec, tvec, end="")
        print(flush=True)

    def ShowResult(self, processor):
        # assert(processor.done)
        self.visualizer.timestamp = processor.timestamp
        self.visualizer.image = processor.image

    def StoreResult(self, processor):
        # assert(processor.done)
        dirName = "aruco"
        
        if not os.path.isdir(dirName):
            os.mkdir(dirName)

        cv2.imwrite("{}/{}_{}.jpg".format(dirName, processor.timestamp, processor.frame), processor.image)

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame; set the current processor going and grab
            # a spare one
            self.frame += 1
            timestamp = str(round(time.time() * 1000))
            if self.processor:
                self.processor.event.set()
            with self.lock:
                if self.pool:
                    if self.pool[-1].done:
                        if (self.args.visualize or self.args.store) and self.pool[-1].corners:
                            for rvec, tvec, crns, id in zip(self.pool[-1].rvecs, self.pool[-1].tvecs, self.pool[-1].corners, self.pool[-1].ids):
                                cv2.aruco.drawAxis(self.pool[-1].image, self.pool[-1].camera_matrix, self.pool[-1].distortion_coeffs, rvec, tvec, 0.176)
                                cv2.putText(self.pool[-1].image, "id: {} x: {:.2f} y: {:.2f} z: {:.2f}".format(id, tvec[0][0], tvec[0][1], tvec[0][2]),
                                    (int(crns[0][0][0]), int(crns[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
                        if self.args.print_results:
                            self.PrintResult(self.pool[-1])
                        if self.args.visualize:
                            self.ShowResult(self.pool[-1])
                        if self.args.store:
                            self.StoreResult(self.pool[-1])
                        self.pool[-1].Reset()
                    self.processor = self.pool.pop()
                    self.processor.frame = self.frame
                    self.processor.timestamp = timestamp
                else:
                    # No processor's available, we'll have to skip
                    # this frame; you may want to print a warning
                    # here to see whether you hit this case
                    if self.args.print_results:
                        print(timestamp, self.frame, flush=True)
                    self.processor = None
        if self.processor:
            self.processor.stream.write(buf)

    def Flush(self):
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

def ParseCommandLineArguments():
    parser = argparse.ArgumentParser(description='Detects ArUco markers and computes their pose.')
    parser.add_argument('-n', '--num-threads', default=1, type=int)
    parser.add_argument('-p', '--print-results', action='store_const', const=True, default=False)
    parser.add_argument('-v', '--visualize', action='store_const', const=True, default=False)
    parser.add_argument('-s', '--store', action='store_const', const=True, default=False)
    parser.add_argument('-m', '--marker-size', required=True, type=float)
    parser.add_argument('-d', '--dictionary', default="DICT_4X4_50", type=str)
    parser.add_argument('-c', '--camera-calibration', default="camera_calibration.p", type=str)
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__":
    args = ParseCommandLineArguments()
    width, height, _, _ = LoadCalibration(args.camera_calibration)

    with picamera.PiCamera(resolution=(width, height), framerate=30) as camera:
        time.sleep(2)
        output = ProcessOutput(args)
        camera.start_recording(output, format='mjpeg')
        try:
            while not output.done:
                camera.wait_recording()
        except KeyboardInterrupt:
            pass
        camera.stop_recording()
        output.Flush()

