import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import cvzone






model=YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
       # print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)   # 括號裡的第一個參考是視窗的名稱 'RGB', 第二個參數是調用上面定義的函數 RGB，設定滑屬移動取得X、y 座標位置。
cap=cv2.VideoCapture('busfinal.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0


area1=[(259,488),(281,499),(371,499),(303,466)]

tracker=Tracker()

counter=[]

while True:    
    ret,frame = cap.read()
    if not ret:   # ret 是布林值，當影片結尾會回傳false代表結束，此時執行break。其餘則count+1，其中此count目的為減低運算偵數，只處理1、4、7、10…
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    
    list=[]  # 要儲存追蹤用的位置，如果只是單純要顯示人體的物件偵測，不需要追蹤位置，則可以把下面的list.append去掉。
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]   #查 這一個index (行)的 row[5]的所儲存的物體名稱id ，並以d做為索引在class_list中查詢類別的名稱。 換句話說，c是一個字串。
        print("c=",c) # yolo 可能抓出person、bus、chair…等物件的名稱，下面必須過濾。
        if 'person' in c:

            list.append([x1,y1,x2,y2])  #追蹤用的，如果沒要追蹤的話，可以直接畫出x1 Y1 X2 Y2的值就結束了。

           # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)
           # cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)   # f'{}' 只是強制字串格式轉換

        #if 'chair' in c:
        #    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        #   cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)  # f'{}' 只是強制字串格式轉換

    bbox_idx=tracker.update(list)   #追蹤用的程式，目的在更新剛剛list儲存的位置。
    #print(bbox_idx)

    for id,rect in bbox_idx.items():
        print(id,rect)  #可以把每個偵測到的人的id，以及各別人物的座標位置印出。
        x3,y3,x4,y4=rect  #另外命名座標，不要跟x1y1x2y2 重覆搞混
        cx=x3 # 人物的bbox的左上角
        cy=y4 # 人物的bbox的左下角
        cv2.circle(frame,(cx,cy),6,(0,255,0),-1)
        cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),2)
        cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)   # f'{}' 只是強制字串格式轉換

    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),3)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:  #waitkey(0) 是凍結，可做逐禎測試。
        break
cap.release()
cv2.destroyAllWindows()
