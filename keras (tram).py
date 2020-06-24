#!/usr/bin/env python
# coding: utf-8

# In[161]:


from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split

caltech_dir = "C://team/train"
categories = ["pomeranian", "shihtzu","Shar Pei","Welsh corgi","Husky","Doberman"]
nb_classes = len(categories)
#강아지 이미지 폴더 경로 지정
image_w = 64
image_h = 64

pixels = image_h * image_w * 3
#이미지 크기 초기화
X = []
y = []

for idx, dog in enumerate(categories):
    
    #one-hot 돌리기.
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

    image_dir = caltech_dir + "/" + dog
    files = glob.glob(image_dir+"/*.jpg")
    print(dog, " 파일 길이 : ", len(files))
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)

        X.append(data)
        y.append(label)

        if i % 700 == 0:
            print(dog, " : ", f)
#그 다음으로 이제 이미지 변환을 해줍니다.

#각 이미지를 가지고 와서 RGB형태로 변환해준 뒤 resize해줍니다.

#그리고 그 값을 numpy 배열로 바꾸고 배열에 추가해주죠.

#동시에 category 값도 넣어줍니다(Y)
X = np.array(X)
y = np.array(y)
#1.000 0.000 0.000 0.000 0.000 0.000 이면 pomeranian

#0.000 1.000 0.000 0.000 0.000 0.000 이면 shihtzu 이런식

X_train, X_test, y_train, y_test = train_test_split(X, y)
xy = (X_train, X_test, y_train, y_test)
np.save("C://numpy_data/multi_image_data.npy", xy)
#이미지 데이터셋을 npy 배열로 저장


# In[162]:


import os, glob, numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
np_load_old = np.load
X_train, X_test, y_train, y_test = np.load('C://numpy_data/multi_image_data.npy',allow_pickle=True)
print(X_train.shape)
print(X_train.shape[0])
#이전에 저장한 npy파일을 읽어온다.


# In[163]:


categories = ["pomeranian", "shihtzu","Shar Pei","Welsh corgi","Husky","Doberman"]
nb_classes = len(categories)

#일반화
X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255


# In[164]:


with K.tf_ops.device('/device:CPU:0'):
    model = Sequential()
    #케라스에서는 층(layer)을 조합하여 모델(model)을 만듭니다. 모델은 (일반적으로) 층의 그래프입니다. 
    #가장 흔한 모델 구조는 층을 차례대로 쌓은 tf.keras.Sequential 모델입니다.
    model.add(Conv2D(32, (3,3), padding="same", input_shape=X_train.shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    #32개의 마스크를 적용, padding = same, input_shape은 64x64, 활성함수는 relu를 사용하였다.
    model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
      
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_dir = './model'
    #nb_classes는 종의 갯수
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    #만일 위의 폴더가 존재하지 않으면 이 이름의 폴더를 만들어 줌
    model_path = model_dir + '/multi_img_classification.model' #모델이 저장되는 경로
    checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=6)
    #val_loss값이 6번 이상 상승되지 자동적으로 멈추게 합니다(오버피팅, overfitting 방지)


# In[165]:


model.summary()
#모델을 10개의 레이어로 가시화함


# In[166]:


history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test), callbacks=[checkpoint, early_stopping])
#batch_size는 32, epochs은 10번씩 돌면서 model을 fit 시켰습니다.


# In[167]:


print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))
#모델의 정확도 출력


# In[168]:


y_vloss = history.history['val_loss'] #검증 손실값
y_loss = history.history['loss'] #훈련 손실값

x_len = np.arange(len(y_loss))

plt.plot(x_len, y_vloss, marker='.', c='red', label='val_set_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='train_set_oss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()
#학습이력을 그래프로 확인


# In[169]:


from PIL import Image
import os, glob, numpy as np
from keras.models import load_model

caltech_dir = "C://multi_img_data/imgs_others_test"
image_w = 64
image_h = 64

pixels = image_h * image_w * 3

X = []
filenames = []
files = glob.glob(caltech_dir+"/*.*")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    filenames.append(f)
    X.append(data)
#테스트할 이미지를 변환
X = np.array(X)
model = load_model('./model/multi_img_classification.model')
#위에서 만든 모델을 load합니다.

#이미지 예측
prediction = model.predict(X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
cnt = 0

for i in prediction:
    pre_ans = i.argmax()  # 예측 레이블
    print(i)
    print(pre_ans)
    pre_ans_str = ''
    if pre_ans == 0: pre_ans_str = "포메라니안"
    elif pre_ans == 1: pre_ans_str = "시츄"
    elif pre_ans == 2: pre_ans_str = "샤페이"
    elif pre_ans == 3: pre_ans_str = "웰시코기"
    elif pre_ans == 4: pre_ans_str = "허스키"        
    else: pre_ans_str = "도베르만"
    if i[0] >= 0.8 : print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
    if i[1] >= 0.8: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"으로 추정됩니다.")
    if i[2] >= 0.8: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
    if i[3] >= 0.8: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
    if i[4] >= 0.8: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
    if i[5] >= 0.8: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
    cnt += 1
    # print(i.argmax()) #가 레이블 [1. 0. 0.] 이런식으로 되어 있는 것을 숫자로 바꿔주는 것.
    # 즉, 나중에 카테고리 데이터 불러와서 카테고리랑 비교를 해서 같으면 맞는거고, 아니면 틀린거로 취급하면 됩니다.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




