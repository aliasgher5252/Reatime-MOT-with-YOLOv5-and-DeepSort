import torch
import numpy as np
import time
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import pandas as pd
import motmetrics as mm
from IPython.display import clear_output


def get_objects_for_frame(dataframe, frame_number):
    # Filter the DataFrame to get rows for the specified frame number
    frame_data = dataframe[dataframe.iloc[:, 0] == frame_number]  # Assuming the first column contains frame numbers

    # Extract tracking IDs and bounding boxes as numpy arrays
    tracking_ids = frame_data.iloc[:, 1].values  # Assuming the second column contains tracking IDs
    bounding_boxes = frame_data.iloc[:, 2:6].values  # Assuming columns 3 to 6 contain bounding box values

    return np.array(tracking_ids), np.array(bounding_boxes)

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
        self.classes=self.model.names
        #print(self.classes)

    def prediction(self,frame,input_shape):
        #resized_frame=cv2.resize(frame,input_shape)
        results=self.model(frame)
        return frame,results.xyxy[0]

    def class_to_label(self,x):
        return self.classes[int(x)]

    def get_bbox(self,frame,confidence,label,input_shape):
        resized_frame, predictions=self.prediction(frame,input_shape)
        detections=[]
        for prediction in predictions:
            #print(prediction.numpy())
            x1, y1, x2, y2, conf, class_id = prediction.cpu().numpy()

            x1, y1, x2, y2, conf, class_id=int(x1), int(y1), int(x2),int(y2), float(conf), int(class_id)
            if class_id==0 and conf>=confidence:
                detections.append(([x1, y1, x2-x1, y2-y1],conf,class_id))
        #print(detections)
        return resized_frame,detections




yolov5_tflite=YoloDetector()

gt_df=pd.read_csv("/content/gt_PET.txt")

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

acc = mm.MOTAccumulator(auto_id=True)
cap = cv2.VideoCapture("/content/ETH-Sunnyday-raw.webm")
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 14, (frame_width,frame_height))
frame_count=1
FPS=[]
while cap.isOpened():
    gt_track_ids,gt_bboxes=get_objects_for_frame(gt_df, frame_count)
    #print(frame_count)
    frame_count+=1
    track_ids=[]
    bboxes=[]

    start=time.time()
    ret, frame = cap.read()

    if not ret:
        break

    resized_frame,detections=yolov5_tflite.get_bbox(frame,confidence=0.5,label='Person',input_shape=(640,640))
    tracks = object_tracker.update_tracks(detections, frame=resized_frame) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        track_ids.append(track_id)

        bbox = track.to_ltrb()
        bboxes.append([bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]])
        #print(bbox)
        cv2.rectangle(resized_frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(resized_frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]-12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, 2)
    distance = mm.distances.iou_matrix(gt_bboxes, bboxes, max_iou=0.5)
    acc.update(gt_track_ids,track_ids,distance)
    #print(acc.events)
    end=time.time()
    fps = 1/(end-start)
    FPS.append(fps)
    cv2.putText(resized_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    out.write(resized_frame)

    #cv2.imshow('Person Tracker', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(f"Average FPS: sum(FPS)/len(FPS)")
mh = mm.metrics.create()
summary = mh.compute_many(
    [acc, acc.events.loc[0:1]],
    metrics=mm.metrics.motchallenge_metrics,
    names=['full', 'part'],
    generate_overall=True
    )

strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)
# Release resources
cap.release()
cv2.destroyAllWindows()

