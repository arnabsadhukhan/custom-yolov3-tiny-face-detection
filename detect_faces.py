import cv2
import numpy as np
import time
import requests



net = cv2.dnn.readNet("trained_weights/yolov3-tiny_custom_final.weights", "yolov3-tiny_custom.cfg")
classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))



window_name = 'Image'
  
#fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 255, 255) 
  
# Line thickness of 2 px 
thickness = 2


done = False
cv2.destroyAllWindows()
#cap = cv2.VideoCapture(0)

#url= 'http://192.168.0.101:8080/shot.jpg'
count = 0
while not done:
    #req = requests.get(url)
    #img_arr = np.array(bytearray(req.content),dtype = np.uint8)
    #frame = cv2.imdecode(img_arr,-1)
    ret,frame = cap.read()
    #frame = cv2.resize(frame,(416,416))
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img = frame
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
   
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indices =  cv2.dnn.NMSBoxes(boxes, confidences, 0.2,0.4)
    for i in class_ids:
        print(classes[i])
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
      
        cv2.rectangle(img, (x, y), (x+w , y+h), (255,0,255), 2)
        x,y,w,h = np.abs([x,y,w,h])*2.5
        #try:
        if True:
            face = frame[int(y):int(y+h+50),int(x):int(x+w)]
            cv2.imshow('faces',face)
                

    cv2.imshow('frame',cv2.resize(img,(416,416)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        done = True

cv2.destroyAllWindows()



