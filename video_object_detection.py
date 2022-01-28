#!/usr/bin/env python3
#https://towardsdatascience.com/yolo-object-detection-with-opencv-and-python-21e50ac599e9
#https://github.com/arunponnusamy/object-detection-opencv
#https://www.codegrepper.com/code-examples/python/python+split+video+into+frames
#https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/

import cv2
import numpy as np
import os
import time

start_time=time.time()
config_arg='input/yolov3.cfg'
weight_arg='input/yolov3.weights'
class_arg='input/yolov3.txt'
COLORS = np.random.uniform(0, 255, size=(len(open(class_arg,'r').readlines()), 3))

def slice_video(video):
	os.system('rm output/video_pics_raw/*') #bash required, probably need a different method to work in Windows
	vidcap = cv2.VideoCapture('input/videos/'+video)
	success,image = vidcap.read()
	count = 0
	while success:
	  cv2.imwrite("output/video_pics_raw/%d.jpg" % count, image)
	  success,image = vidcap.read()
	  print(video+' - Reading frame: %d ' % count)
	  count += 1

def get_output_layers(net):    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h,classes,COLORS):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
def parse_data_set(pic_arg,COLORS,vid):
	os.system('rm output/video_pics_processed/*') #bash required, probably need a different method to work in Windows
	image = cv2.imread('output/video_pics_raw/'+pic_arg)
	Width = image.shape[1]
	Height = image.shape[0]
	scale = 0.00392
	classes = None
	with open(class_arg, 'r') as f:
		classes = [line.strip() for line in f.readlines()]
	net = cv2.dnn.readNet(weight_arg, config_arg)
	blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(get_output_layers(net))
	class_ids = []
	confidences = []
	boxes = []
	conf_threshold = 0.5
	nms_threshold = 0.4
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.5:
			    center_x = int(detection[0] * Width)
			    center_y = int(detection[1] * Height)
			    w = int(detection[2] * Width)
			    h = int(detection[3] * Height)
			    x = center_x - w / 2
			    y = center_y - h / 2
			    class_ids.append(class_id)
			    confidences.append(float(confidence))
			    boxes.append([x, y, w, h])
	indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
	for i in indices:
		box = boxes[i]
		x = box[0]
		y = box[1]
		w = box[2]
		h = box[3]
		draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), classes ,COLORS)
	print(vid+' - Processing: '+pic_arg)
	cv2.imwrite('output/video_pics_processed/'+pic_arg, image)

def form_video(vid):
	img_array = []
	Size=(0,0)
	for i in range(0,len(os.listdir('output/video_pics_processed/'))-1):
		img = cv2.imread('output/video_pics_processed/'+str(i)+'.jpg')
		height, width, layers = img.shape
		Size = (width,height)
		img_array.append(img)
	out = cv2.VideoWriter('output/videos/'+vid[:-3]+'avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, Size) #works for file extensions with length of 3 chars, (e.g. mp4, mov, mkv, etc.)
	for i in range(len(img_array)):
		print(vid+' - Writing frame: '+str(i))
		out.write(img_array[i])
	out.release()
	os.system('rm output/video_pics_processed/*') #bash required, probably need a different method to work in Windows
	os.system('rm output/video_pics_raw/*') #bash required, probably need a different method to work in Windows
	print('cleaned')
	
for vid in os.listdir('input/videos'):
	slice_video(vid)
	for pic in os.listdir('output/video_pics_raw'):
		parse_data_set(pic,COLORS,vid)
	form_video(vid)
print('Run time: '+str(time.time()-start_time), ' seconds')
