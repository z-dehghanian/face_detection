
# coding: utf-8

# In[39]:


from PIL import Image
from numpy import  array
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage
from sklearn import linear_model

x_train = []
y_train = []

x_test = []
y_test = []

images = []

with open('faces\\train.txt') as f:
    content = f.readlines()

with open('faces\\test.txt') as f:
    test_content = f.readlines()

avg = np.zeros((50, 50))

for line in content:
    path = line.split(' ')
    image_arr = array(Image.open(path[0]))
    x_train.append(image_arr)
    y_train.append(path[1])
    avg += image_arr

for line in test_content:
    path = line.split(' ')
    image_arr = array(Image.open(path[0]))
    x_test.append(image_arr)
    y_test.append(path[1])
    

avg /= len(x_train)
#print(avg)
#toimage(avg).show()
#toimage(x_train[1]).show()

for i in range(len(x_train)):
    x_train[i] = x_train[i] - avg
for i in range(len(x_test)):
    x_test[i] = x_test[i] - avg    
#print ("avg :")
#print (avg)
#toimage(x_train[1]).show()

u, s, v = np.linalg.svd(x_train, full_matrices=True)
print(v.shape)
for i in range(0,10):
   # toimage(v[i]).show()
    plt.imshow(v[i], cmap='gray')
    plt.show()




