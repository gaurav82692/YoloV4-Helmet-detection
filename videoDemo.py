import cv2
import time
import numpy as np
from PIL import Image

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.5

cap = cv2.VideoCapture("hel3.mp4")
prevTime = 0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

size = (width, height)

result = cv2.VideoWriter('output2.avi', 
                         cv2.VideoWriter_fourcc(*'XVID'),
                         10, size)

while cap.isOpened(): 
    ret, frame = cap.read()
    img = np.array(frame)
    
    class_names = []
    with open("coco.names", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]

    
    colors = np.random.uniform(0,255,size=(len(class_names),3))
    net = cv2.dnn.readNet("custom-yolov4-tiny-detector_best.weights", "custom-yolov4-tiny-detector.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(640, 640), scale=1/255, swapRB=True)
    
    classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    
    
    
    for (classid, score, box) in zip(classes, scores, boxes):
        #color=colors[classid[0]]
        score=score[0]
        score=score*100
        score= "{:.2f}".format(score)
        #color=colors[classid[0]]
        print(score)
        img2=img[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
        #crop = image[y:y+h, x:x+w]
        x=int((box[2])/2)
        y=int((box[3])/3.5)
        ifa = Image.fromarray(img2)
        color = ifa.getpixel((x,y))
        label = "%s : %s" % (class_names[classid[0]], score+'%')
        print(label)
        if class_names[classid[0]]=='Head':
            cv2.rectangle(img, box,(0,0,255), 2)
            cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 2)
        else:
            cv2.rectangle(img, box,color, 2)
            cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)
        #result.write(frame)


    cv2.imshow('frame',img)
    result.write(img)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        result.release()
        cv2.destroyAllWindows()
        break 