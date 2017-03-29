import numpy as np
import sys
from keras.models import load_model
from PIL import Image
from keras import backend as K

image_path = sys.argv[1] # [1] indicate the second argument in put on terminal 
im = Image.open(image_path)

#convert input image ro grayscale image
im = im.convert('L') 
img_rows, img_cols= 28,28
im = im.resize((img_rows, img_cols),Image.ANTIALIAS) #resize the image
im = np.asarray(im)
print('[INFO]',im.shape)

n_sample = 1
n_channel = 1

image = im/np.max(im).astype('float32') #normalise input
test_image1=image

if K.image_data_format()== 'channels_first': #for tensorflow backends
	test_image1=image.reshape((n_sample,img_rows,img_cols,n_channel)) # reshape it to our input placeholder shape
else: #for theano backends
	test_image1=image.reshape((n_channel,img_rows,img_cols,n_sample)) # reshape it to our input placeholder shape

# Load keras model here
model= load_model('my_model1.h5')

predictions = model.predict_proba(test_image1)
print('\n',predictions)
predicted_class= np.argmax(predictions)
print ('\n[INFO] Predicted class :',format(predicted_class))
print('\n')
