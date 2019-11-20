import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import SGD
import glob
import shutil

#モデル、学習した重み、クラス、入力切り出し犬画像、出力latte画像
file_name='vgg16_dog_fine'
model_name = 'vgg16_dog_fine.h5'
folder=["dog","latte"]
dir_input_name = "yolooutput"
dir_output_name="latte"

#モデル読み込み、重み読み込み
json_string=open(file_name+'.json').read()
model=model_from_json(json_string)
model.load_weights(model_name)

#image  array size
img_size = (224,224)

#File type
file_type  = 'jpg'

#load images and image to array
img_list = glob.glob('./' + dir_input_name + '/*.' + file_type)
temp_img_array_list = []
img_name=[]
for img in img_list:
    img_name.append(img)
    temp_img = load_img(img,grayscale=False,target_size=(img_size))
    temp_img_array = img_to_array(temp_img) /255
    temp_img_array_list.append(temp_img_array)
temp_img_array_list = np.array(temp_img_array_list)

#クラス分類実行
img_pred=model.predict(temp_img_array_list)
list=np.argmax(img_pred,axis=1)

#latte画像だけ保存
b=0
for a in list:
    print(folder[a],img_name[b])
    if folder[a]=="latte":
        shutil.copy(img_name[b],dir_output_name+"/"+str(b)+".jpg")
    b=b+1
