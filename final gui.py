#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:



import tensorflow as tf 
from keras.preprocessing import image
import numpy as np
import os
import pandas

# image folder
folder_path = '/DATA/strawberrytest/'
# path to model
model_path = '/DATA/fruitclassification/Fruitmodel.h5'
img_width, img_height = 200, 200

# load the trained model
model = tf.keras.models.load_model(model_path)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[12]:


#GUI File
from PIL import Image               # to load images
from IPython.display import display # to display images
import os

    
from tkinter import *
  
# loading Python Imaging Library 
from PIL import ImageTk, Image 
  
# To get the dialog box to open when required  
from tkinter import filedialog

imgs = []

def openfilename(): 
  
    # open file dialog box to select image 
    # The dialogue box has a title "Open" 
    filename = filedialog.askopenfilename(title ='"pen') 
    return filename

def open_img(imgs):
    # Select the Imagename  from a folder  
    x = openfilename() 
  
    # opens the image 
    img = Image.open(x) 
      
    # resize the image and apply a high-quality down sampling filter 
    img = img.resize((200, 200), Image.ANTIALIAS) 
  
    imgs.append(img)
    # PhotoImage class is used to add image to widgets, icons etc 
    img = ImageTk.PhotoImage(img)
   
    # create a label 
    panel = Label(root, image = img) 
      
    # set the image as img  
    panel.image = img 
    panel.pack()
# Create a windoe 
root = Tk() 

# Set Title as Image Loader 
root.title("Image Loader") 

# Set the resolution of window 
root.geometry("550x300") 

# Allow Window to be resizable 
root.resizable(width = True, height = True) 

# Create a button and place it into the window using grid layout 
btn = Button(root, text ='open image', command = open_img(imgs))
btn.pack()


T = Text(root, height=2, width=30)
T.pack()
T.insert(END, "")
# test_image = image.load_img(imgs[0], target_size = (200, 200))
test_image = image.img_to_array(imgs[0])
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
t=np.argmax(result , axis=1)
output = ''
if t<1:
    output = 'ripe'
elif 1<=t<2:
    output = 'damaged'
else:
    output = 'unripe'

T.insert(END, output)




from tkinter.messagebox import *

# def show_answer():

#     ans.set(Ans)
#     Entry(main,  text = "%s" %(ans) ).grid(row=2, column=1)
root.mainloop() 


# In[9]:


from PIL import Image               # to load images
from IPython.display import display # to display images
import os
pil_im = Image.open('/DATA/Strawberrytest/s1.jpg')
display(pil_im)

import cv2
import numpy
img = cv2.imread('/DATA/Strawberrytest/s1.jpg')
avgcolorperrow = numpy.average(img, axis=0)
avgcolor = numpy.average(avgcolorperrow, axis=0)
print(avgcolor)


# In[13]:


#Normal Classification

from PIL import Image               # to load images
from IPython.display import display # to display images
import os
pil_im = Image.open('/DATA/Strawberrytest/2.jpg')
display(pil_im)

import tensorflow as tf 
from keras.preprocessing import image
import pandas
# Part 3 - Making new predictions
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.preprocessing import image
# path to model
model_path = '/DATA/fruitclassification2/Fruitmodel3.h5'

# load the trained model
model = tf.keras.models.load_model(model_path)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
test_image = image.load_img('/DATA/strawberrytest/2.jpg', target_size = (200, 200))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
t=np.argmax(result , axis=1)
print(t)


# In[ ]:




