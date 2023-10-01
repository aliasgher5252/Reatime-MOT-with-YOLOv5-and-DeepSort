import torch
import numpy as np
import time
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import argparse
COCO_CLASSES = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorcycle': 3,
    'airplane': 4,
    'bus': 5,
    'train': 6,
    'truck': 7,
    'boat': 8,
    'traffic light': 9,
    'fire hydrant': 10,
    'stop sign': 11,
    'parking meter': 12,
    'bench': 13,
    'bird': 14,
    'cat': 15,
    'dog': 16,
    'horse': 17,
    'sheep': 18,
    'cow': 19,
    'elephant': 20,
    'bear': 21,
    'zebra': 22,
    'giraffe': 23,
    'backpack': 24,
    'umbrella': 25,
    'handbag': 26,
    'tie': 27,
    'suitcase': 28,
    'frisbee': 29,
    'skis': 30,
    'snowboard': 31,
    'sports ball': 32,
    'kite': 33,
    'baseball bat': 34,
    'baseball glove': 35,
    'skateboard': 36,
    'surfboard': 37,
    'tennis racket': 38,
    'bottle': 39,
    'wine glass': 40,
    'cup': 41,
    'fork': 42,
    'knife': 43,
    'spoon': 44,
    'bowl': 45,
    'banana': 46,
    'apple': 47,
    'sandwich': 48,
    'orange': 49,
    'broccoli': 50,
    'carrot': 51,
    'hot dog': 52,
    'pizza': 53,
    'donut': 54,
    'cake': 55,
    'chair': 56,
    'couch': 57,
    'potted plant': 58,
    'bed': 59,
    'dining table': 60,
    'toilet': 61,
    'tv': 62,
    'laptop': 63,
    'mouse': 64,
    'remote': 65,
    'keyboard': 66,
    'cell phone': 67,
    'microwave': 68,
    'oven': 69,
    'toaster': 70,
    'sink': 71,
    'refrigerator': 72,
    'book': 73,
    'clock': 74,
    'vase': 75,
    'scissors': 76,
    'teddy bear': 77,
    'hair drier': 78,
    'toothbrush': 79
}
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-source','--path', help='Provide the source of the video you want to infer', required=True)
parser.add_argument('-labels','--labels', help='List of objects you want to track e.g ["person", "car"]', required=True,type=str,nargs='+')
parser.add_argument('-name','--name', help='Name of your output file', required=True)
parser.add_argument('-save','--save', help='Directory of output file', required=True)
args = vars(parser.parse_args())
VIDEO_PATH = args['path']
LABELS=args['labels']
OUTPUT_NAME=args['name']
SAVE_PATH=args['save'] + '/'+ OUTPUT_NAME +".avi"
print(VIDEO_PATH,LABELS,OUTPUT_NAME)
class YoloDetector():

    def __init__(self,model_path=None):
        if model_path!=None:
            self.model = torch.hub.load('.', 'custom', path=model_path, source='local')
            print("Model Successfully Loaded...")
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m',pretrained=True)

    
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using Device: {self.device}")

        self.model=self.model.to(self.device)
    
    def prediction(self,frame):
        #resized_frame=cv2.resize(frame,input_shape)
        results=self.model(frame)
        return frame,results.xyxy[0]
    
    def get_classIDs(self,labels):
        return [COCO_CLASSES[label] for label in labels]
    
    def get_bbox(self,frame,confidence,labels):
        resized_frame, predictions=self.prediction(frame)
        class_ids=self.get_classIDs(labels)
        detections=[]
        for prediction in predictions:
            x1, y1, x2, y2, conf, class_id = prediction.cpu().numpy()
            
            x1, y1, x2, y2, conf, class_id=int(x1), int(y1), int(x2),int(y2), float(conf), int(class_id)
            if (class_id in class_ids) and conf>=confidence:
                detections.append(([x1, y1, x2-x1, y2-y1],conf,class_id))
        return resized_frame,detections
    



yolov5=YoloDetector()

object_tracker =  DeepSort(max_age=5,
                n_init=2,
                nms_max_overlap=1.0,
                max_cosine_distance=0.3,
                nn_budget=None,
                override_track_class=None,
                embedder="clip_RN50",
                half=True,
                bgr=False,
                embedder_gpu=True,
                embedder_model_name=None,
                embedder_wts=None,
                polygon=False,
                today=None)

cap = cv2.VideoCapture(VIDEO_PATH)

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(SAVE_PATH, fourcc, 20, (frame_width,frame_height))
FPS=[]
print('STARTED TRACKING...')
while cap.isOpened():
    start=time.time()
    ret, frame = cap.read()

    if not ret:
        break

    resized_frame,detections=yolov5.get_bbox(frame,confidence=0.5,labels=LABELS)
    tracks = object_tracker.update_tracks(detections, frame=resized_frame) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
    
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()
        #print(bbox)
        cv2.rectangle(resized_frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(resized_frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]-12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, 2)

    end=time.time()
    fps = 1/(end-start)
    FPS.append(fps)
    cv2.putText(resized_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Person Tracker", resized_frame)
    out.write(resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(f"Average FPS: {sum(FPS)/len(FPS)}")
# Release resources
cap.release()
cv2.destroyAllWindows()




