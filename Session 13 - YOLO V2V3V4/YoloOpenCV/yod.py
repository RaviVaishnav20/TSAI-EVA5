import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
# print(classes)

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255, size=(len(classes), 3))
#Load image
img = cv2.imread("m3.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.2)
height, width, channels = img.shape

#Detecting Objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# for b in blob:
#     for n, img_blob in enumerate(b):
#         cv2.imshow(str(n), img)

net.setInput(blob)
outs = net.forward(outputlayers)

boxes = []
confidences = []
class_ids = []
# showing informations on the screen
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            #Object detection
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2)
            # Rectangele coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            # cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
print(len(boxes))
indexs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexs)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexs:
        x, y, w, h = boxes[i]
        color = colors[i]
        labels = str(classes[class_ids[i]])
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
        cv2.putText(img,labels, (x, y-2), font, 2,  color,4)

filename = 'savedImage.jpg'

cv2.imwrite(filename, img)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()