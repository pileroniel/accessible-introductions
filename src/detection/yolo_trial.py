from ultralytics import YOLO
from pathlib import Path

prediction_path= Path("outputs/detections")
data_path = Path("data/raw")
yolo_model = YOLO("yolo26n.pt")
results = yolo_model(f"{data_path}/ron_arsenal.jpg", classes=[0])

"Draw YOLO Prediction bounding boxes and display it to the user. Save it in the detection folder "
def draw_bounding_boxes(results):
  for result in results:
    boxes = result.boxes # boxes object for bounding box outputs
    masks = result.masks # masks object for segmentation masks outputs
    keypoint = result.keypoints # keypoints object for pose outputs
    probs = result.probs # probs object for classification outputs 
    obb = result. obb # oriented boxes for OBB outputs
    result.show() # display to screen
    result.save (filename=f"{prediction_path}/ron_arsenal.jpg")
    


if __name__ == "__main__":
  print(len(results))
  draw_bounding_boxes(results)
  # for r in results[0]:
  #   print(f"Detected {len(r)} objects in image")
  #   print(r)
  #   print(r.names, r.masks, r.new, r.numpy, r.obb, r.plot, r.save_crop)
  


