#using harcasscade classifier-pretrained model
import cv2
cap=cv2.VideoCapture(0)
#create a classifier object which works on facial data
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
while True:
    ret,frame=cap.read()
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if ret==False:
        continue
    faces=face_cascade.detectMultiScale(gray_frame,1.3,5)#returns a tuple of starting x coor,ycoor and width and hight
    #2 para-scaling factor and number of neihgbours
     #scalefactor- how much the img size is reduced at each img scale
     #haar cascade works upon fixed size of image that is why we need to reduce the size,scale factor 1.3 means reduce the size by 30 percent till the desired size is reached(orsimilar)
     # higer number of neihbors result in less detection but better quality,preferred 3 to 6
    
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)#color in BGR
    cv2.imshow("Video frame",frame)
    key_pressed=cv2.waitKey(1)&0xFF
    if key_pressed==ord('q'):
        break
cap.release()
cv2.destroyAllWindows