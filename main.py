import json

import cv2
import numpy as np

MIN_SCORE = 0.5
SIZE = 640
STRIDES = [8, 16, 32]
FONT = cv2.FONT_HERSHEY_SIMPLEX

with open("labels.json") as file:
    coco = json.load(file)['coco']
NAMES =[coco[key] for key in sorted(coco, keys=int)]

net = cv2.dnn.readNetFromONNX("detector.onnx")

cells= [SIZE // stride for stride in STRIDES]
GRID = np.concatenate([np.indices((n,n))[::-1].reshape(2,-1).T for n in cells])
STRIDES = np.concatenate([np.full((n*n,1),s) for n,s in zip(cells, STRIDES)])


def detect(frame):
    height, width = frame.shape[:2]
    scale =(min(SIZE / height, SIZE / width))
    small= cv2.resize(frame, (round(width* scale), round(height*scale)))
    canvas = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
    canvas[:small.shape[0], :small.shape[1]] = small

    net.setInput(cv2.dnn.blobFromImage(canvas))
    output = net.forward()[0]


    centers = (output[:, :2] + GRID) * STRIDES
    sizes = np.exp(output[:, 2:4]) * STRIDES
    scores = output[:, 4:5] * output[:, 5:]
    class_ids = np.argmax(scores, axis=1)
    confidences = scores.max(axis=1)

    boxes = (np.hstack([centers - sizes / 2, sizes]) /scale)
    keep =cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), class_ids.tolist(), MIN_SCORE, 0.5)
    return[(NAMES[class_ids[i]], boxes[i]) for i in keep]

video = cv2.VideoCapture(0)

while True:
    ok, frame = video.read()
    if not ok:
        break
    for name, (x, y,w, h) in detect(frame):
       hsv =np.unit8([[[NAMES.index(name)*29% 180, 255, 255]]])
       colour= tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0])
       cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
    cv2.imshow("live detect", frame)
    if cv2.waitKey(1) & 255 == ord("q"):
        break

video.release()
cv2.destroyAllWindows()