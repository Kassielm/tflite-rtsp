import sys, getopt
import numpy as np
from time import time
import os
import cv2
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

## Import tflite runtime
import tflite_runtime.interpreter as tf #Tensorflow_Lite


## Read Environment Variables
USE_HW_ACCELERATED_INFERENCE = True

## The system returns the variables as Strings, so it's necessary to convert them where we need the numeric value
if os.environ.get("USE_HW_ACCELERATED_INFERENCE") == "0":
    USE_HW_ACCELERATED_INFERENCE = False

MINIMUM_SCORE = float(os.environ.get("MINIMUM_SCORE", default = 0.55))

CAPTURE_DEVICE = os.environ.get("CAPTURE_DEVICE", default = "/dev/video0")

CAPTURE_RESOLUTION_X = int(os.environ.get("CAPTURE_RESOLUTION_X", default = 640))

CAPTURE_RESOLUTION_Y = int(os.environ.get("CAPTURE_RESOLUTION_Y", default = 480))

CAPTURE_FRAMERATE = int(os.environ.get("CAPTURE_FRAMERATE", default = 30))

STREAM_BITRATE = int(os.environ.get("STREAM_BITRATE", default = 2048))

## Helper function to draw bounding boxes
def draw_bounding_boxes(img,labels,x1,x2,y1,y2,object_class):
    # Define some colors to display bounding boxes
    box_colors=[(254,153,143),(253,156,104),(253,157,13),(252,204,26),
             (254,254,51),(178,215,50),(118,200,60),(30,71,87),
             (1,48,178),(59,31,183),(109,1,142),(129,14,64)]

    text_colors=[(0,0,0),(0,0,0),(0,0,0),(0,0,0),
             (0,0,0),(0,0,0),(0,0,0),(255,255,255),
            (255,255,255),(255,255,255),(255,255,255),(255,255,255)]

    cv2.rectangle(img,(x2,y2),(x1,y1),
                box_colors[object_class%len(box_colors)],2)
    cv2.rectangle(img,(x1+len(labels[object_class])*10,y1+15),(x1,y1),
                box_colors[object_class%len(box_colors)],-1)
    cv2.putText(img,labels[object_class],(x1,y1+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                text_colors[(object_class)%len(text_colors)],1,cv2.LINE_AA)

## Media factory that runs inference
class InferenceDataFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(InferenceDataFactory, self).__init__(**properties)

        # Setup frame counter for timestamps
        self.number_frames = 0
        self.duration = (1.0 / CAPTURE_FRAMERATE) * Gst.SECOND  # duration of a frame in nanoseconds

        # Create opencv Video Capture
        self.cap = cv2.VideoCapture(f'v4l2src device={CAPTURE_DEVICE} extra-controls="controls,horizontal_flip=1,vertical_flip=1" ' \
                                    f'! video/x-raw,width={CAPTURE_RESOLUTION_X},height={CAPTURE_RESOLUTION_Y},framerate={CAPTURE_FRAMERATE}/1 ' \
                                    f'! videoconvert primaries-mode=fast n-threads=4 ' \
                                    f'! video/x-raw,format=BGR ' \
                                    f'! appsink', cv2.CAP_GSTREAMER)
        
        # Create factory launch string
        self.launch_string = f'appsrc name=source is-live=true format=GST_FORMAT_TIME ' \
                             f'! video/x-raw,format=BGR,width={CAPTURE_RESOLUTION_X},height={CAPTURE_RESOLUTION_Y},framerate={CAPTURE_FRAMERATE}/1 ' \
                             f'! videoconvert primaries-mode=fast n-threads=4 ' \
                             f'! video/x-raw,format=I420 ' \
                             f'! x264enc bitrate={STREAM_BITRATE} speed-preset=ultrafast tune=zerolatency threads=4 ' \
                             f'! rtph264pay config-interval=1 name=pay0 pt=96 '
        
        # Setup execution delegate, if empty, uses CPU
        if(USE_HW_ACCELERATED_INFERENCE):
            delegates = [tf.load_delegate("/usr/lib/libvx_delegate.so")]
        else:
            delegates = []

        # Load the Object Detection model and its labels
        with open("labels.txt", "r") as file:
            self.labels = file.read().splitlines()

        # Create the tensorflow-lite interpreter
        self.interpreter = tf.Interpreter(model_path="best_float32.tflite",
                                          experimental_delegates=delegates)

        # Allocate tensors.
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size=self.input_details[0]['shape'][1]


    # Funtion to be ran for every frame that is requested for the stream
    def on_need_data(self, src, length):

        if self.cap.isOpened():
            # Read the image from the camera
            ret, image_original = self.cap.read()

            if ret:
                # Resize the image to the size required for inference
                height1=image_original.shape[0]
                width1=image_original.shape[1]
                image=cv2.resize(image_original,
                                (self.input_size,int(self.input_size*height1/width1)),
                                interpolation=cv2.INTER_NEAREST)
                height2=image.shape[0]
                scale=height1/height2
                border_top=int((self.input_size-height2)/2)
                image=cv2.copyMakeBorder(image,
                                border_top,
                                self.input_size-height2-border_top,
                                0,0,cv2.BORDER_CONSTANT,value=(0,0,0))

                # Set the input tensor
                input=np.array([cv2.cvtColor(image, cv2.COLOR_BGR2RGB)],dtype=np.uint8)

                # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
                # This is valid for mobilenet architecture and may not be valid for other architectures 
                if self.input_details[0]['dtype'] == np.float32:
                    input = (np.float32(input) - 127.5) / 127.5

                self.interpreter.set_tensor(self.input_details[0]['index'], input)

                # Execute the inference
                t1=time()
                self.interpreter.invoke()
                t2=time()
                
                # Check output layer name to determine if this model was created with TF2 or TF1,
                # because outputs are ordered differently for TF2 and TF1 models. This is valid
                # for mobilenet architecture and may not be valid for other architectures
                outname = self.output_details[0]['name']

                if ('StatefulPartitionedCall' in outname): # This is a TF2 model
                    locations_idx, classes_idx, scores_idx, detections_idx = 1, 3, 0, 2
                else: # This is a TF1 model
                    locations_idx, classes_idx, scores_idx, detections_idx = 0, 1, 2, 3

                # Check the detected object locations, classes and scores.
                locations = (self.interpreter.get_tensor(self.output_details[locations_idx]['index'])[0]*width1).astype(int)
                locations[locations < 0] = 0
                locations[locations > width1] = width1
                classes = self.interpreter.get_tensor(self.output_details[classes_idx]['index'])[0].astype(int)
                scores = self.interpreter.get_tensor(self.output_details[scores_idx]['index'])[0]
                n_detections = self.interpreter.get_tensor(self.output_details[detections_idx]['index'])[0].astype(int)

                # Draw the bounding boxes for the detected objects
                img = image_original
                for i in range(n_detections):
                    if (scores[i]>MINIMUM_SCORE):
                        y1 = locations[i,0]-int(border_top*scale)
                        x1 = locations[i,1]
                        y2 = locations[i,2]-int(border_top*scale)
                        x2 = locations[i,3]
                        draw_bounding_boxes(img,self.labels,x1,x2,y1,y2,classes[i])

                # Draw the inference time
                cv2.rectangle(img,(0,0),(130,20),(255,0,0),-1)
                cv2.putText(img,"inf time: %.3fs" % (t2-t1),(0,15),cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255,255,255),1,cv2.LINE_AA)

                # Create and setup buffer
                data = GLib.Bytes.new_take(img.tobytes())
                buf = Gst.Buffer.new_wrapped_bytes(data)
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp
                self.number_frames += 1

                # Emit buffer
                retval = src.emit('push-buffer', buf)
                if retval != Gst.FlowReturn.OK:
                    print(retval)

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)


class RtspServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(RtspServer, self).__init__(**properties)
        # Create factory
        self.factory = InferenceDataFactory()

        # Set the factory to shared so it supports multiple clients
        self.factory.set_shared(True)

        # Add to "inference" mount point. 
        # The stream will be available at rtsp://<board-ip>:8554/inference
        self.get_mount_points().add_factory("/inference", self.factory)
        self.attach(None)

def main():
    Gst.init(None)
    server = RtspServer()
    loop = GLib.MainLoop()
    loop.run()

        

if __name__ == "__main__":
    main()
