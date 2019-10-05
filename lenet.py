#Convolutional Neural Network (Copied from PyImageSearch)
#Kieran Hobden
#20-Sep-'19

#This document is not original work but a pre-made script with minor changes made for education purposes


from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses, activation="relu", weightsPath=None):
        #Initialise the model
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)

        #Update the input shape if channels first is used
        if K.image_data_format() == "channels_first":
            inputShape = (numChannels, imgRows, imgCols)

        #Define the first set of CONV => ACTIVATION => POOL layers
        model.add(Conv2D(20, 5, padding="same", input_shape=inputShape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        #Define the second set of CONV => ACTIVATION => POOL layers
        model.add(Conv2D(50, 5, padding="same"))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        #Define the first FC => ACTIVATION layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))

        #Define the second FC layer
        model.add(Dense(numClasses))

        #Lastly, define the soft-max classifier
        model.add(Activation("softmax"))

        #If the model was pre-trained and a weights path is supplied, load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        #Return the constructed network architecture
        return model
