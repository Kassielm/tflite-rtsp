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
        
        # Define some colors to display bounding boxes, one for each class
        self.colors = [ (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for _ in self.labels]


        # Create the tensorflow-lite interpreter
        self.interpreter = tf.Interpreter(model_path="best_float32_edgetpu.tflite",
                                          experimental_delegates=delegates)

        # Allocate tensors.
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input size and scaling ratios for drawing boxes
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
        self.height_ratio = CAPTURE_RESOLUTION_Y / self.input_height
        self.width_ratio = CAPTURE_RESOLUTION_X / self.input_width

        # Thresholds for post-processing
        self.iou_threshold = 0.5 # Intersection Over Union threshold
        self.threshold = MINIMUM_SCORE # Confidence score threshold


    # Funtion to be ran for every frame that is requested for the stream
    def on_need_data(self, src, length):
        if self.cap.isOpened():
            # Read the image from the camera
            ret, frame = self.cap.read()

            if ret:
                # Resize and convert to RGB
                input_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_image = cv2.resize(input_image_rgb, (self.input_width, self.input_height))
                
                # --- CORREÇÃO APLICADA AQUI ---
                # Converte a imagem para float32 ANTES de qualquer outra operação
                input_data = np.expand_dims(input_image.astype(np.float32), axis=0)
                
                # Normaliza a imagem se o modelo esperar float (ex: de -1 a 1 ou 0 a 1)
                # A maioria dos modelos YOLO espera valores normalizados de 0 a 1.
                input_data = input_data / 255.0

                # Se o modelo for INT8, o delegate da NPU lida com a quantização a partir do float.
                # O importante é que o tipo enviado corresponda ao que a assinatura de entrada do modelo pede.
                # O erro confirma que ele espera FLOAT32.
                
                # Set the input tensor and run inference
                t1=time()
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                t2=time()
                
                # --- LÓGICA DE PÓS-PROCESSAMENTO DO YOLOv8 (mantém-se igual) ---

                # 1. Get the single output tensor
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                # De-quantize if necessary
                output_details = self.output_details[0]
                if output_details['dtype'] == np.int8 or output_details['dtype'] == np.uint8:
                    output_scale, output_zero_point = output_details['quantization']
                    output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

                # 2. Transpose the output from [7, 8400] to [8400, 7]
                outputs = np.transpose(output_data)

                # 3. Build lists of boxes, scores and class IDs
                boxes = []
                scores = []
                class_ids = []

                for row in outputs:
                    # Get box coordinates [cx, cy, w, h] and class probabilities
                    box_coords = row[:4]
                    class_probs = row[4:]

                    # Find the class with the highest probability
                    class_id = np.argmax(class_probs)
                    max_score = class_probs[class_id]

                    # Filter out detections with low confidence
                    if max_score > self.threshold:
                        scores.append(max_score)
                        class_ids.append(class_id)
                        
                        # Convert box from [cx, cy, w, h] to [x1, y1, x2, y2] and scale it
                        cx, cy, w, h = box_coords
                        x1 = int((cx - w / 2) * self.width_ratio)
                        y1 = int((cy - h / 2) * self.height_ratio)
                        x2 = int((cx + w / 2) * self.width_ratio)
                        y2 = int((cy + h / 2) * self.height_ratio)
                        boxes.append([x1, y1, x2, y2])

                # 4. Apply Non-Maximum Suppression (NMS) to remove redundant boxes
                boxes_for_nms = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    boxes_for_nms.append([x1, y1, x2 - x1, y2 - y1])

                if len(boxes_for_nms) > 0:
                    indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores, self.threshold, self.iou_threshold)
                else:
                    indices = []

                # 5. Draw the final bounding boxes on the original frame
                for i in indices:
                    box = boxes[i]
                    x1, y1, x2, y2 = box
                    label = self.labels[class_ids[i]]
                    score = scores[i]
                    color = self.colors[class_ids[i]]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw the inference time
                cv2.rectangle(frame,(0,0),(130,20),(0,0,255),-1)
                cv2.putText(frame,"inf time: %.3fs" % (t2-t1),(0,15),cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255,255,255),1,cv2.LINE_AA)

                # Create and setup buffer for streaming
                data = GLib.Bytes.new_take(frame.tobytes())
                buf = Gst.Buffer.new_wrapped_bytes(data)
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp
                self.number_frames += 1

                # Emit buffer
                src.emit('push-buffer', buf)

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