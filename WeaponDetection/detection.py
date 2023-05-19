from ultralytics import YOLO
import cv2
import cvzone
import math
import winsound

# cap=cv2.VideoCapture('../videos/gun2.mp4')
# cap=cv2.VideoCapture('../videos/people.mp4')

cap=cv2.VideoCapture(0)
model = YOLO("../bestweapon.pt")

classNames=['pistol']
count=0
myColor=(0,255,0)
#bgr
while True:
    success,img=cap.read()
    print(type(img))
    print(type(img))
    print(type(img))
    print(type(img))
    results=model(img,stream=True)

    for r in results:
        boxes=r.boxes
        for box in boxes:
            myColor = (0,255, 0)
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1
            cv2.rectangle(img,(x1,y1),(x2,y2),myColor,3)
            conf=math.ceil(box.conf[0]*100)/100 #confidence
            # print(conf)
            cls=int(box.cls[0])
            if(classNames[cls]=='pistol' and conf>0.7):
                myColor=(0,0,255)
                # cvzone.putTextRect(img, f'{classNames[cls]}', (max(0, x1), max(35, y1)), scale=1, thickness=1,
                #                    colorB=myColor, colorT=(255, 255, 255), colorR=myColor)
                filename="Image"+str(count)+".jpg"
                count+=1
                cv2.imwrite(filename,img)
                # winsound.Beep(1445, 100)

    cv2.imshow("Image",img)
    cv2.waitKey(1)