from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import CSVLogger

#class=[dog,latte]
n_categories=2
batch_size=8
#画像があるディレクトリの名前
train_dir='train'
validation_dir='validation'
#保存する、学習した重み、モデルデータの名前
file_name='vgg16_dog_fine'
number_of_sumple=301

#学習済みモデルを読み込む
base_model=VGG16(weights='imagenet',include_top=False,
                 input_tensor=Input(shape=(224,224,3)))

#層を追加
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
prediction=Dense(n_categories,activation='softmax')(x)
model=Model(inputs=base_model.input,outputs=prediction)

#モデルの前半は学習しない
for layer in base_model.layers[:15]:
    layer.trainable=False

#モデル準備
model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#モデル表示
model.summary()

#モデル保存
json_string=model.to_json()
open(file_name+'.json','w').write(json_string)

#学習データ準備、水増し

train_datagen=ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen=ImageDataGenerator(rescale=1.0/255)

train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator=validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

#学習実行
hist=model.fit_generator(train_generator,
                         epochs=50,
                         verbose=1,
                        steps_per_epoch=number_of_sumple // batch_size,
                         validation_steps=1,
                         validation_data=validation_generator,
                         callbacks=[CSVLogger(file_name+'.csv')])

#学習した重み保存
model.save(file_name+'.h5')
