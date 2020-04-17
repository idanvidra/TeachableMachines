import cv2
import numpy as np
from PIL import Image, ImageOps
import tensorflow.keras
from butter.mas.api import ClientFactory
import time

# Connect to robot
ip = '0.0.0.0'
butterTcpClient = ClientFactory().getClient(ip, protocol='tcp')


#Load the saved model
model = tensorflow.keras.models.load_model("D:\\Progrmming\\Milab\\Projects\\Teachable Machines\\Pikaboo\\Keras\\keras_model.h5")
video = cv2.VideoCapture(0)

while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into 224x224 because we trained the model with this image
        #size.
        size = (224, 224)
        im = ImageOps.fit(im, size, Image.ANTIALIAS)
        #im = im.resize((224,224))
        img_array = np.asarray(im)
        #img_array = np.array(im)

        # Normalize the image
        normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 224x224x3 into 1x224x224x3 
        #img_array = np.expand_dims(img_array, axis=0)
        normalized_image_array = np.expand_dims(normalized_image_array, axis=0)

        #Calling the predict method on model to predict 'me' on the image
        #prediction = model.predict(img_array)[0][0]
        prediction = model.predict(normalized_image_array)[0][0]


        
        #if prediction is 0, which means I am missing on the image, then show the frame in gray color.
        if prediction < 0.5:
                
                butterTcpClient.playAnimation("Nod_45")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                time.sleep(3)
                # Closed
        else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Open
                

        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break

video.release()
cv2.destroyAllWindows()