import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
import os
from PIL import Image

#(1) 데이터 
paths = glob.glob('PetImages/Cat/*.jpg')

for f in paths:
    try:
        img = Image.open(f)
        img_resize = img.resize((32,32))
        print("PetImages/Cat/{}".format(f.split('\\')[-1]))
        img_resize.save("PetImages/Cat/{}".format(f.split('\\')[-1]))
    except:
        pass
#print(paths) #[] 
# print(len(paths)) #25000
# paths = np.random.permutation(paths) 