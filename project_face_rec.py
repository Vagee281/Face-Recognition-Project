import cv2
import numpy as np

#init camera
cap=cv2.VideoCapture(0)
#face detection
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
skip=0
face_data=[]
dataset_path='./data/'

file_name=input("Enter the name of the person: ")
while True:
    ret,frame=cap.read()

    if ret==False:
        continue
    
 
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces=face_cascade.detectMultiScale(frame,1.3,5)#list of tuples
    #we have to sort the faces on the basis of largest area and strore the largest one
    # since the x,y,w,h is stored in the form of tuple we will do indexing and use the widht and height to cal area=h*w
    if len(faces)==0:
        continue
    faces=sorted(faces,key=lambda f:f[2]*f[3])
    #pick the last face because it is the largest one a.t area
    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        #extract(crop out the required face):Region of interest
        #in frame by convention 1st axis is y axis
        offset=10
        #global face_section
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        #store every 10th face
        skip+=1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))#how many face data I have stored till now
        


        
    cv2.imshow("Frame",frame)
    cv2.imshow("face section",face_section)


   
    key_pressed=cv2.waitKey(1)&0xFF
    if key_pressed==ord('q'):
        break
#convert our face list array into a numpy array
#number of rows must be same as number of faces
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)

print("Data successfully saved at "+dataset_path+file_name+'.npy')


    
cap.release()
cv2.destroyAllWindows()



