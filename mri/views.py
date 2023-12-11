from django.shortcuts import render
from django.core.files.base import ContentFile
from PIL import Image
import io
import cv2
import numpy as np
import keras
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import base64
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

def home(request):
    return render(request,'home.html')

def about(request):
    return render(request,'about.html')

def credits(request):
    return render(request,'credits.html')

def result(request):
    error = 0
    result = None
    img_byte_arr = None
    res = None
    if request.method == 'POST':
        try:
            model = keras.models.load_model(BASE_DIR / 'mri/model.h5')
            img = request.FILES.get('image')
            img = Image.open(img)
            realimg = img
            img_byte_arr = io.BytesIO()
            # define quality of saved array
            realimg.save(img_byte_arr, format='JPEG', subsampling=0, quality=100)
            # converts image array to bytesarray
            img_byte_arr = base64.b64encode(img_byte_arr.getvalue()).decode('UTF-8')
            img = np.asarray(img)
            cpyimg = img
            dim = (224,224)
            img = cv2.resize(img, dim,interpolation = cv2.INTER_AREA)
            X = image.img_to_array(img)
            X = np.expand_dims(X, axis=0)
            if X.shape[3] == 1:
                img = cpyimg
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                img = cv2.resize(img, dim,interpolation = cv2.INTER_AREA)
                X = image.img_to_array(img)
                X = np.expand_dims(X, axis=0)
            X = preprocess_input(X)
            result = model.predict(X)
            for r in result.tolist()[0]:
                print(r*100)
            result = result.argmax()
            category = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
            result = category[int(result)]
            error = 1
        except:
            error = 2
    return render(request, 'result.html', {'result': result, 'img':img_byte_arr, 'error':error})