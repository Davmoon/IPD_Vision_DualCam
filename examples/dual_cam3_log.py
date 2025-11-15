import degirum as dg
import degirum_tools
import degirum_tools.streams as dgstreams
from picamera2 import Picamera2
import cv2
import time
import sys

# --- [����] ---
inference_host_address = "@local"
zoo_url = "../models"
token = '' 

# --- [ī�޶� ���ʷ�����] ---
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

# --- [����Ʈ �˸� Gizmo] ---
class NotificationGizmo(dgstreams.Gizmo):
    def __init__(self, camera_name):
        super().__init__([(10,)]) 
        self.camera_name = camera_name
        self.frame_count = 0

    def run(self):
        print(f"[{self.camera_name}] �˸��� ��� ��...", flush=True)
        
        for result_wrapper in self.get_input(0):
            # 1. ����� ���� ���� �ʱ�ȭ
            inference_result = None

            # 2. 'data'�� ������� Ȯ��
            if hasattr(result_wrapper.data, 'results'):
                inference_result = result_wrapper.data
            else:
                # 3. 'data'�� �ƴ϶�� 'meta'(��Ÿ������) ����Ʈ�� ������ ã��
                # (result_wrapper.meta�� StreamMeta ��ü�̰�, ���ο� _meta_list�� ����)
                # ������ API�� find_last�� get�� ���� �� ������, �±׸� �𸣹Ƿ� ����Ʈ�� ���� ��ȸ
                try:
                    # StreamMeta Ŭ������ ���� ����Ʈ�� ����
                    meta_list = result_wrapper.meta._meta_list
                    for item in meta_list:
                        if hasattr(item, 'results'):
                            inference_result = item
                            break
                except:
                    pass

            # 4. ����� ã�Ҵٸ� �м� ����
            if inference_result and inference_result.results:
                for obj in inference_result.results:
                    label = obj.get('label', '')
                    score = obj.get('score', 0) * 100
                    
                    # 'scooter'�� ���Ե� �󺧸� ���
                    if 'scooter' in label:
                        print(f"\n?? [{self.camera_name}] �߰�! ����: '{label}' ({score:.1f}%)", flush=True)

            # 5. �۵� Ȯ�ο� �� ��� (60�����Ӹ���)
            self.frame_count += 1
            if self.frame_count % 60 == 0:
                print(".", end="", flush=True)
            
            # 6. ���� �ܰ�� ������ ����
            self.send_result(result_wrapper)

# --- [���� ����] ---
configurations = [
    { "model_name": "scooter_model", "source" : '0', "display_name": "cam0" },
    { "model_name": "scooter_model", "source" : '1', "display_name": "cam1" },
]

# --- [�� �ε�] ---
models = [
    dg.load_model(cfg["model_name"], inference_host_address, zoo_url, token)
    for cfg in configurations
]

# --- [Gizmo ����] ---
sources = [dgstreams.IteratorSourceGizmo(picamera_generator(int(cfg["source"]))) for cfg in configurations]
detectors = [dgstreams.AiSimpleGizmo(model) for model in models]
notifiers = [NotificationGizmo(cfg["display_name"]) for cfg in configurations]
display = dgstreams.VideoDisplayGizmo(
    [cfg["display_name"] for cfg in configurations], show_ai_overlay=True, show_fps=True
)

# --- [���������� ����] ---
pipeline = (
    (source >> detector for source, detector in zip(sources, detectors)),
    (detector >> notifier >> display[di] 
     for di, (detector, notifier) in enumerate(zip(detectors, notifiers)))
)

# --- [����] ---
print("�ý��� ����...", flush=True)
dgstreams.Composition(*pipeline).start()