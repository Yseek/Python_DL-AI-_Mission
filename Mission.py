import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
import os
from PIL import Image

#사진 편집 (고양이)
# paths = glob.glob('PetImages/Cat/*.jpg')
# for f in paths:
#     try:
#         img = Image.open(f)
#         img_resize = img.resize((32,32))
#         print("PetImages/Cat/{}".format(f.split('\\')[-1]))
#         img_resize.save("PetImages/Cat/{}".format(f.split('\\')[-1]))
#     except:
#         pass

# # 사진 편집 (강아지)
# paths = glob.glob('PetImages/Dog/*.jpg')
# for f in paths:
#     try:
#         img = Image.open(f)
#         img_resize = img.resize((32,32))
#         print("PetImages/Dog/{}".format(f.split('\\')[-1]))
#         img_resize.save("PetImages/Dog/{}".format(f.split('\\')[-1]))
#     except:
#         pass

#(1) 데이터
paths = glob.glob("PetImages/*/*.jpg")
#print(paths) #[] 
print(len(paths)) #1301
paths = np.random.permutation(paths) 

independent = np.array([plt.imread(paths[i]) for i in range(len(paths))])
dependent = np.array([paths[i].split('\\')[-2] for i in range(len(paths))])
print(independent.shape, dependent.shape) # (1301, 32, 32, 3) (1301,)

# print("dependent[0:5]", dependent[0:5])
# plt.imshow(independent[0]) #위와 일치 확인 
# plt.show()
dependent = pd.get_dummies(dependent)

#(2) 모델
X = tf.keras.layers.Input(shape=[32,32,3]) #3차원으로 변경

H = tf.keras.layers.Conv2D(6, kernel_size=5,activation='swish')(X) #컨벌루션 mask1 적용
H = tf.keras.layers.MaxPool2D()(H) #풀링1
H = tf.keras.layers.Conv2D(16, kernel_size=5, activation='swish')(H) #컨벌루션 mask2 적용
H = tf.keras.layers.MaxPool2D()(H) #풀링2

H = tf.keras.layers.Flatten()(H) #평탄화
H = tf.keras.layers.Dense(120, activation='swish')(H) #변경
H = tf.keras.layers.Dense(84, activation='swish')(H) #변경
Y = tf.keras.layers.Dense(2, activation="softmax")(H) 
model = tf.keras.models.Model(X,Y)  
model.compile(loss='categorical_crossentropy', metrics='accuracy')

#모델유형 분석
model.summary() 

#(3) 학습 
model.fit(independent, dependent, epochs=100) #loss: 0.1485 - accuracy: 0.9528

#(4) 검증( 예측값 : 원본값 )
print("< 판단값 >")
pre = model.predict(independent[0:5])
print(pd.DataFrame(pre).round(2)) # A, D, C, H, H

print("< 실제값 >") 
print(dependent[0:5]) # A, D, C, H, H