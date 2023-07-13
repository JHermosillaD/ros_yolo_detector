#!/usr/bin/env python
import numpy as np
import roslib
import rospy
import time

from custom_uv_msgs.msg import ImageBoundingBox
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO
import torch
import cv2

class yolo_detection:  
  model = None
  category_index = None

  def __init__(self):
    print('Loading model...', end='')
    start_time = time.time()
    
    if torch.cuda.is_available(): 
      dev = "cuda:0"
      print("using cuda ..")
    else: 
      dev = "cpu"
      print("using cpu ..")
      
    self.device = torch.device(dev)
    self.model = YOLO('/home/jhermosilla/Downloads/yolov8n.pt')
    self.model.to(self.device)  
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    self.bridge = CvBridge()
    self.image_pub = rospy.Publisher("/human_image/compressed", CompressedImage, queue_size=1)
    self.image_sub = rospy.Subscriber("/kinect2/qhd/image_color_rect", Image, self.callbackDetection, queue_size=1)
    self.box_pub = rospy.Publisher("/humanBBox", ImageBoundingBox, queue_size=1)

  def callbackDetection(self,ros_data):
    try:
      colorImage = self.bridge.imgmsg_to_cv2(ros_data, "bgr8")
    except CvBridgeError as e:
      print(e)

    results = self.model.predict(colorImage, verbose=False, device="cuda:1")
    for result in results:
      boxes = result.boxes.cpu().numpy()
      for box in boxes:
        if (box.cls[0].item() == 0):
          r = box.xyxy[0].astype(int)
          cv2.rectangle(colorImage, r[:2], r[2:], (0, 0, 255), 2)

    box = results[0].boxes
    if (box.xyxy.shape[0]>0):
      pixel_box = box.xyxy[0].tolist()
      box_width = round(pixel_box[2] - pixel_box[0])
      box_height = round(pixel_box[3] - pixel_box[1])
      bbox_msg = ImageBoundingBox()
      bbox_msg.header.frame_id = "base_link"
      bbox_msg.header.stamp = rospy.Time.now()
      bbox_msg.center.u = round(pixel_box[0] + box_width/2)
      bbox_msg.center.v = round(pixel_box[1] + box_height/2)
      bbox_msg.width = box_width
      bbox_msg.height = box_height
      bbox_msg.cornerPoints[0].u = round(pixel_box[0])
      bbox_msg.cornerPoints[0].v = round(pixel_box[1])
      bbox_msg.cornerPoints[1].u = round(pixel_box[0] + box_width)
      bbox_msg.cornerPoints[1].v = round(pixel_box[1])
      bbox_msg.cornerPoints[2].u = round(pixel_box[0] + box_width)
      bbox_msg.cornerPoints[2].v = round(pixel_box[1] + box_height)
      bbox_msg.cornerPoints[3].u = round(pixel_box[0])
      bbox_msg.cornerPoints[3].v = round(pixel_box[1] + box_height)
      self.box_pub.publish(bbox_msg)

    try:
      self.image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(colorImage))
    except CvBridgeError as e:
      print(e)
      
def main():    
  rospy.init_node('yolo_detection', anonymous=True)
  ic = yolo_detection()
  
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
    main()
