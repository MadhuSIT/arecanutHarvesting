from django.shortcuts import render
import numpy as np
from django.conf import settings
from django.core.files.storage import default_storage
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import os

model_path = os.path.join(os.path.dirname(__file__), 'models', 'arecanut.h5')
model = tf.keras.models.load_model(model_path)
if model:
    print('model loaded')
else:
    print("model not found")
    exit(0)

dic={0:'GREEN',1:'RIPE'}
size = (224,224)


def index(request):
    detec = None
    if request.method == 'POST':
        file = request.FILES['imageFile']
        file_name = default_storage.save(file.name,file)
        file_path = default_storage.path(file_name)

        image = load_img(file_path,target_size=(224,224))
        numpy_array = img_to_array(image)
        image_ndims = np.expand_dims(numpy_array,axis=0)
        image_ndims = image_ndims/255
        # print(type(image_ndims),image_ndims.ndim,image_ndims.itemsize,sep="\t")
       
        result = model.predict(image_ndims)
        prob = result.max()
        predclass = np.argmax(result)
        if prob>0.85 and predclass==1:
                detec=dic[predclass]
                #engine.say(detec)
                #engine.runAndWait()
        elif prob >0.80 and predclass==0:
            detec=dic[predclass]
        else:
            detec="SemiRipe"



        # test_image = cv2.resize(test_image, size)
        # test_image = image.img_to_array(test_image)
        # test_image = np.expand_dims(test_image, axis = 0)
        # test_image = test_image/255

    return render(request,'index.html',{'result':detec})
    
