import numpy as np
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Activation
from keras.optimizers import Adam

timestep = 60
np.random.seed(10)

num_class = 5
train_sample_per_class = 10

# For Bandwith Prediction:
# PredictionType = 0
# TrainWithAllData = True

#For Duration Prediction:
# PredictionType = 1
# num_class = 4
# TrainWithAllData = True

#For Traffic Class Prediction:
PredictionType = 2
TrainWithAllData = False



trainData = np.load("trainData.npy")
trainlabel = np.load("trainLabel.npy")
trainData = trainData[:, -timestep*2:]
trainlabel = trainlabel[:, -timestep*2:]
trainlabel = trainlabel.astype(int)

testData = np.load("testData.npy")
testLabel = np.load("testLabel.npy")
testData = testData[:, -timestep*2:]
testLabel = testLabel[:, -timestep*2:]
testLabel = testLabel.astype(int)


for i in range(trainlabel.shape[0]):
    #Categorizing Bandwidth
    if trainlabel[i, 0] < 10000:
        trainlabel[i, 0] = 1
    elif trainlabel[i, 0] < 50000:
        trainlabel[i, 0] = 2
    elif trainlabel[i, 0] < 100000:
        trainlabel[i, 0] = 3
    elif trainlabel[i, 0] < 1000000:
        trainlabel[i, 0] = 4
    else:
        trainlabel[i, 0] = 5
    #Categorizing Duration
    if trainlabel[i, 1] < 10:
        trainlabel[i, 1] = 1
    elif trainlabel[i, 1] < 30:
        trainlabel[i, 1] = 2
    elif trainlabel[i, 1] < 60:
        trainlabel[i, 1] = 3
    else:
        trainlabel[i, 1] = 4

for i in range(testLabel.shape[0]):
    #Categorizing Bandwidth
    if testLabel[i, 0] < 10000:
        testLabel[i, 0] = 1
    elif testLabel[i, 0] < 50000:
        testLabel[i, 0] = 2
    elif testLabel[i, 0] < 100000:
        testLabel[i, 0] = 3
    elif testLabel[i, 0] < 1000000:
        testLabel[i, 0] = 4
    else:
        testLabel[i, 0] = 5
    #Categorizing Duration
    if testLabel[i, 1] < 10:
        testLabel[i, 1] = 1
    elif testLabel[i, 1] < 30:
        testLabel[i, 1] = 2
    elif testLabel[i, 1] < 60:
        testLabel[i, 1] = 3
    else:
        testLabel[i, 1] = 4


trainlabel = trainlabel[:, PredictionType]
testLabel = testLabel[:, PredictionType]


train_size = trainlabel.shape[0]
Y_train = np.zeros((train_size, num_class))
Y_train[np.arange(train_size),trainlabel[:]-1] = 1

test_size = testLabel.shape[0]
Y_test = np.zeros((test_size, num_class))
Y_test[np.arange(test_size),testLabel[:]-1] = 1

trainData = trainData.reshape((trainData.shape[0], timestep, 2))
testData = testData.reshape((testData.shape[0], timestep, 2))

if not TrainWithAllData:
    class_counter = np.zeros((num_class))
    x_train = np.zeros((num_class*train_sample_per_class, timestep, 2))
    y_train = np.zeros((num_class*train_sample_per_class, 5))
    j = 0
    for i in range(train_size):
        class_id = trainlabel[i] - 1
        if class_counter[class_id] < train_sample_per_class:
            x_train[j] = trainData[i]
            y_train[j] = Y_train[i]
            j += 1
            class_counter[class_id] += 1

    trainData = x_train
    Y_train = y_train

def base_model():

    model_input = Input(shape=(timestep,2))

    x = Conv1D(32, 3, activation='relu')(model_input)
    x = Conv1D(32, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=(2))(x)

    x = Conv1D(64, 3, activation='relu')(x)
    x = Conv1D(64, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=(2))(x)

    x = Conv1D(128, 3, activation='relu')(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=(2))(x)

    x = Flatten()(x)

    x3 = Dense(256)(x)
    x3 = Activation('relu')(x3)
    x3 = Dense(256)(x3)
    x3 = Activation('relu')(x3)
    output3 = Dense(num_class, activation='softmax', name='Class')(x3)

    model = Model(inputs=model_input, outputs=[output3])
    opt = Adam(clipnorm = 1.)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

model = base_model()

model.fit(trainData, Y_train, validation_data = (testData, Y_test),
          batch_size = 64, epochs = 30, verbose = True, shuffle = True)
