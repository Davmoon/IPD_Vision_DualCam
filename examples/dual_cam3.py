import degirum as dg
import degirum_tools
import degirum_tools.streams as dgstreams
from picamera2 import Picamera2
import cv2
import time

# inference_host_address = "@cloud"
inference_host_address = "@local"

# choose zoo_url
#zoo_url = "degirum/models_hailort"
zoo_url = "../models"

# set token
#token = degirum_tools.get_token()
token = '' # leave empty for local inference

def picamera_generator(index):
    picam2 = Picamera2(index)
    config = picam2.create_preview_configuration(main={"size": (640, 480)}) 
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)
    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            yield frame_bgr
    finally:
        picam2.stop()

class NotificationGizmo(dgstreams.Gizmo):
    def __init__(self, camera_name):
        super().__init__([(10,)])
        self.camera_name = camera_name
        self.frame_count = 0

    def run(self):
        #print(f"[{self.camera_name}]")
        
        for result_wrapper in self.get_input(0):
            inf_result = None

            if hasattr(result_wrapper.data, 'result'):
                inf_result = result_wrapper.data
            else:
                try:
                    meta_list = result_wrapper.meta._meta_list
                    for item in meta_list:
                        if hasattr(item, 'results'):
                            inf_result = item
                            break
                except:
                    pass

            if inf_result and inf_result.results:
                for obj in inf_result.results:
                    label = obj.get('label', '')
                    score = obj.get('score', 0) * 100

                    if 'scooter' in label:
                        print(f"\n[{self.camera_name}] '{label}' ({score:.1f}%)", flush=True)

            self.frame_count += 1
            if self.frame_count % 60 == 0:
                print(".", end="", flush=True)
            
            self.send_result(result_wrapper)

# Define the configurations for video file and webcam
configurations = [
    {
        "model_name": "scooter_model",
        "source" : '0',
        "display_name": "cam0",
    },
    {
        "model_name": "scooter_model",
        "source" : '1',
        "display_name": "cam1",
    },
]


# load models
models = [
    dg.load_model(cfg["model_name"], inference_host_address, zoo_url, token)
    for cfg in configurations
]

# define gizmos
sources = [dgstreams.IteratorSourceGizmo(picamera_generator(int(cfg["source"]))) for cfg in configurations]
detectors = [dgstreams.AiSimpleGizmo(model) for model in models]
notifiers = [NotificationGizmo(cfg["display_name"]) for cfg in configurations]
display = dgstreams.VideoDisplayGizmo(
    [cfg["display_name"] for cfg in configurations], show_ai_overlay=True, show_fps=True
)

# create pipeline
pipeline = (
    (source >> detector for source, detector in zip(sources, detectors)),
    (detector >> notifiers >> display[di] for di, (detector, notifiers) in enumerate(zip(detectors, notifiers))),
)

# start composition
dgstreams.Composition(*pipeline).start()