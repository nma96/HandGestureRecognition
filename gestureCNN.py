from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras import backend as K
K.set_image_data_format('channels_first')

import numpy as np
import os
import theano

from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import cv2

# input image dimensions
row_pixel_count, column_pixel_count = 200, 200

# number of channels
# For grayscale use 1 value and for color images use 3 (R,G,B channels)
img_channels = 1

## NOTE: If you change this then dont forget to change Labels accordingly
number_of_classes = 5

# Total number of convolutional filters to use
number_of_filters = 32
# Max pooling
number_of_pool = 2
# Size of convolution kernel
number_of_convolution_kernel = 3

#%%

prev = -1

Weight_File_Name = ["ori_4015imgs_weights.hdf5"]
#Weight_File_Name = ["newWeight.hdf5"]
#Weight_File_Name = ["newWeight_2.hdf5"]

# outputs
#Labels = ["OK", "NOTHING","PEACE", "PUNCH", "STOP"]
Labels = ["", "", "V", "A", "B"]
#Labels = ["A", "V"]

X = ""


def load_Neural_Network():
	global get_output
	CNN_model = Sequential()
	
	CNN_model.add(Conv2D(number_of_filters, (number_of_convolution_kernel, number_of_convolution_kernel),
						padding='valid',
						input_shape=(img_channels, row_pixel_count, column_pixel_count)))
	convout1 = Activation('relu')
	CNN_model.add(convout1)
	CNN_model.add(Conv2D(number_of_filters, (number_of_convolution_kernel, number_of_convolution_kernel)))
	convout2 = Activation('relu')
	CNN_model.add(convout2)
	CNN_model.add(MaxPooling2D(pool_size=(number_of_pool, number_of_pool)))
	CNN_model.add(Dropout(0.5))

	CNN_model.add(Flatten())
	CNN_model.add(Dense(128))
	CNN_model.add(Activation('relu'))
	CNN_model.add(Dropout(0.5))
	CNN_model.add(Dense(number_of_classes))
	CNN_model.add(Activation('softmax'))
	CNN_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
	
	# CNN_model summary
	CNN_model.summary()
	# CNN_model conig details
	CNN_model.get_config()

	#Load pretrained weights
	fname = Weight_File_Name[int(0)]
	print ("loading ", fname)
	CNN_model.load_weights(fname)
	layer = CNN_model.layers[11]
	get_output = K.function([CNN_model.layers[0].input, K.learning_phase()], [layer.output,])
	
	
	return CNN_model

# This function does the guessing work based on input images
def predict(CNN_model, img, frame):
	global Labels, get_output, prev, X
	#Load image and flatten it
	image = np.array(img).flatten()
	
	# reshape it
	image = image.reshape(img_channels, row_pixel_count,column_pixel_count)
	
	# float32
	image = image.astype('float32') 
	
	# normalize it
	image = image / 255
	
	# reshape for NN
	rimage = image.reshape(1, img_channels, row_pixel_count, column_pixel_count)
	
	prob_array = get_output([rimage, 0])[0]

	d = {}
	i = 0
	
	for items in Labels:
		d[items] = prob_array[0][i] * 100
		if(d[items]>70.0):
			#print(d[items],items)
			cv2.putText(frame,'Labels:'+items,(30,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2,1)
			if(prev!=i):
				# X+=items
				# cv2.putText(frame,'Labels:'+X,(30,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2,1)
				print ( prev," ",i)
				prev=i
				if i == 4:
					jump = ''' osascript -e 'tell application "System Events" to key code 49' '''
					#jump = ''' osascript -e 'tell application "System Events" to key down (49)' '''
					os.system(jump)
					print("print")	
		i += 1    
	return 1