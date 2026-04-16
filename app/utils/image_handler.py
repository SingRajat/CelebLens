import cv2 #it will deal with image processing
from io import BytesIO
import numpy as np

def process_image(image_file):  #the image file is basically the uploaded image by the user
    in_memory_file=BytesIO() #BytesIO is storing the image in memory for some time
    image_file.save(in_memory_file)

    image_bytes=in_memory_file.getvalue() #this is the being used to get image data from bytes format
    nparr=np.frombuffer(image_bytes,np.uint8) #converting byte data into numpy array


    img=cv2.imdecode(nparr,cv2.IMREAD_COLOR) #this will be used for  decoding the image from bytes to opencv image format

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #this is gray scale image which is best for face detection

    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #loading a pretrained frontal face detection model

    faces=face_cascade.detectMultiScale(gray,1.1,5) #detecting all the faces in the image


    if len(faces)==0:
        return image_bytes,None

    largest_faces=max(faces,key=lambda r:r[2]*r[3]) #extracting the largest face by area or max coverage of a face in a image

    (x,y,w,h)=largest_faces #extracting the coordinates of the largest face

    pad=int(w*0.15)
    y_start=max(0, y-pad)
    y_end=min(img.shape[0],y+h+pad)
    x_start=max(0,x-pad)
    x_end=min(img.shape[1],x+w+pad)

    clean_cropped_face=img[y_start:y_end,x_start:x_end]


    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3) #creating a rectangle around the largest face

    #Encoding the image with rectangle into JPEG format
    is_success,buffer=cv2.imencode('.jpg',img)
    is_success, clean_buffer=cv2.imencode('.jpg',clean_cropped_face)

    return buffer.tobytes(), clean_buffer.tobytes(), largest_faces #returning the image with rectangle and the largest face coordinates
