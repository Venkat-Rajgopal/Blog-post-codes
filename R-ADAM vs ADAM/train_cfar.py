# script to compare the performance of Adam vs R-Adam
# usage
# python3 train_cfar_v1.py --optimizer radam --plot radam.png

import argparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# import keras relates modules
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  MaxPooling2D, Conv2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from RAdam import RAdamOptimizer
from tensorflow.python.keras.utils import to_categorical
import os 
os.environ['TF_KERAS'] = '1'
from keras_radam import RAdam

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
# -------------------------------------------------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--optimizer", type=str, default="adam", choices=["adam", "radam"], help="type of optmizer")
ap.add_argument("-c", "--nepochs", type=int, required=True, help="no of epochs")				
ap.add_argument("-p", "--plot", type=str, required=True, help="path to output training plot")
args = vars(ap.parse_args())


# Choice of optimizers
if args["optimizer"] == "adam":

	print("Using Adam")
	opt = Adam(lr=1e-3)

else:
    print("Using Rectified-Adam")
    opt = RAdam(total_steps=5000, warmup_proportion=0.1, min_lr=1e-5)


# load train and test dataset and convert to cat
def load_dataset():
	(trainX, trainY), (testX, testY) = cifar10.load_data()

	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)

	return trainX, trainY, testX, testY
 
# scale pixels
def scale_pixels(train, test):
	# convert integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')

	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0

	return train_norm, test_norm


# structure a simple cnn 
def define_model(opt = opt):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.2))
	model.add(Dense(10, activation='softmax'))

	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def plot_results(history, plotname, n_epochs, subtitle):

	n = np.arange(0, n_epochs)
	plt.style.use('seaborn')
	plt.figure()
	plt.plot(n, history.history['loss'], label='train_loss')
	plt.plot(n, history.history['val_loss'], label='val_loss')
	plt.plot(n, history.history['accuracy'], label='train_acc')
	plt.plot(n, history.history['val_accuracy'], label='val_acc')
	plt.title('Training and Validation Loss/Accuracy: CIFAR_10' + ':' + subtitle)
	plt.legend()
	plt.xlabel('Epoch #')
	plt.ylabel('Loss/Accuracy')

	plt.savefig(plotname)
	plt.close()


# run model
def evaluate_model(opt = opt, plotname = args['plot'], n_epochs = args['nepochs']):
	# load dataset and scale
	trainX, trainY, testX, testY = load_dataset()
	trainX, testX = scale_pixels(trainX, testX)

	# define model and fit 2d model
	model = define_model(opt = opt)
	history = model.fit(trainX, trainY, epochs=n_epochs, batch_size=64, validation_data=(testX, testY), verbose=1)
	print(history.history)
	
	# evaluate model
	_, acc = model.evaluate(testX, testY, verbose=1)
	print('> %.3f' % (acc * 100.0))

	# plot learning curves
	plot_results(history, plotname = plotname, n_epochs = n_epochs, subtitle = args['optimizer'])
 
# Call model function
evaluate_model()