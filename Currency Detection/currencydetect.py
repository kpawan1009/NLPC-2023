import pyttsx3
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import winsound
import numpy as np

videolink='../videos/currency2.mp4'
cap=cv2.VideoCapture(videolink) #for video
# vcap=cv2.VideoCapture(0)
# cap=vcap
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
vcap = cv2.VideoCapture(videolink)  # 0=camera
width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

model=YOLO('../yolo-weights/detectcurrency.pt')

classNames = ['Rs10', 'Rs100', 'Rs20', 'Rs200', 'Rs2000', 'Rs50', 'Rs500']

# linev

limitsup=[2,2,width-10,2]
limitsdown=[2,height-10,width-10,height-10]

totalCountsup=[]
totalCountsdown=[]
exitid=[]

while(True):
    success , img=cap.read()
    imgRegion=cv2.bitwise_and(img,img)
    results=model(imgRegion,stream=True)
    detections=np.empty((0,5))
    for r in results:
        boxes=r.boxes
        for box in boxes:

            # Bounding Box

            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            # print(x1,y1,x2,y2)

            # confidence

            confidence=box.conf[0]
            confidence=math.ceil((box.conf[0]*100))/100;
            print(confidence)
            cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, rt=5)
            cls=int(box.cls[0])
            currentClass=classNames[cls]
            print(currentClass)
            if(currentClass=='Rs10' or currentClass=='Rs20' or currentClass=='Rs100' or currentClass=='Rs200' or currentClass=='Rs2000' or currentClass=='Rs50' or currentClass=='Rs500' and confidence > 0.6):
                if(currentClass=='Rs200'):
                    if(confidence<.7):
                        continue

                text_speech = pyttsx3.init()
                text_speech.say(currentClass)
                text_speech.runAndWait()
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9,rt=5)
                cvzone.putTextRect(img,f'{classNames[cls]} {confidence}',(50,50),scale=2,thickness=1,offset=3)
                currentArray=np.array([x1,y1,x2,y2,confidence])
                detections=np.vstack((detections,currentArray))

    resultsTracker=tracker.update(detections)
    cv2.line(img,(limitsup[0],limitsup[1]),(limitsup[2],limitsup[3]),(0,0,255),4)
    cv2.line(img,(limitsdown[0],limitsdown[1]),(limitsdown[2],limitsdown[3]),(0,0,255),4)

    for result in resultsTracker:
        x1,y1,x2,y2,id=result
        x1, y1, x2, y2 ,id= int(x1), int(y1), int(x2), int(y2),int(id)
        print(result)
        cvzone.cornerRect(img,(x1,y1,x2-x1,y2-y1),l=9,rt=2,colorR=(255,0,0))
        cvzone.putTextRect(img, f'{currentClass}', (max(0, x1), max(35, y1)), scale=2, thickness=1,
                           offset=3)
        #     finding center
        cx,cy=x1+(x2-x1)//2,y1+(y2-y1)//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limitsup[1]<cy<limitsup[1]+100:
            if totalCountsup.count(id)==0 and totalCountsdown.count(id)>0 and exitid.count(id)==0:
                print("Hello Guys")
                totalCountsdown.remove(id)
                exitid.append(id)
            elif totalCountsup.count(id)==0 and totalCountsdown.count(id)==0 and exitid.count(id) == 0:
                totalCountsup.append(id)


        if limitsdown[3]-100<cy<limitsdown[3]:
            if totalCountsup.count(id)>0 and totalCountsdown.count(id) == 0 and exitid.count(id) == 0:
                # print("Hello Guys")
                totalCountsup.remove(id)
                exitid.append(id)
            elif totalCountsdown.count(id) == 0 and  totalCountsup.count(id)==0 and exitid.count(id) == 0:
                totalCountsdown.append(id)

    people=len(totalCountsup)+len(totalCountsdown)-2*len(exitid)


    cv2.imshow("Image",img)
    # cv2.imshow("Imageregion", imgRegion)
    cv2.waitKey(1)