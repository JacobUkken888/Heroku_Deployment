#import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import cv2
import keras.preprocessing.image as K_image

app = Flask(__name__)
classifier = pickle.load(open('face_detection.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/identify',methods=['POST'])
def predict():
    image = cv2.imread('/content/drive/My Drive/DeepLearning/FaceDetector/test_image/IMG_20200514_184205.jpg')
    image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    faces = fd.detectMultiScale(image_gray, scaleFactor = 1.3, minNeighbors=1, minSize=(500,500))

    for x, y, w, h in faces:
        face = image[y:y+h, x:x+w]
        cv2.imwrite('color_img.jpg', face)
        img=K_image.load_img('color_img.jpg',target_size=(512,512))
        img_array=K_image.img_to_array(img)
        if classifier.predict_classes(img_array[None,:,:,:])[0]==0:
            face = "Jacob"        
        else:
            face = "Ryan"

    return render_template('index.html', identified_face='This is: $ {}'.format(face))


if __name__ == "__main__":
    app.run(debug=True)


    




        


    
