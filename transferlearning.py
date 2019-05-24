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
retrain_sample_per_class = 10
pretrain_test_sample_per_class = 10


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

trainmask = np.zeros((trainlabel.shape[0],256))

retrain_size = num_class*retrain_sample_per_class
pretrain_test_size = num_class*pretrain_test_sample_per_class
train_size = trainlabel.shape[0]
pretrain_size = train_size - retrain_size - pretrain_test_size

x_pretrain = np.zeros((pretrain_size, timestep*2))
y_pretrain = np.zeros((pretrain_size, 2))

x_test_pretrain = np.zeros((pretrain_test_size, timestep*2))
y_test_pretrain = np.zeros((pretrain_test_size, 2))

x_retrain = np.zeros((retrain_size, timestep*2))
y_retrain = np.zeros((retrain_size))


class_counter = np.zeros((num_class))
j = 0   #Counter for pretrain train set
l = 0   #counter for pretrain test set
k = 0   #counter for retrain train set
for i in range(train_size):
    class_id = trainlabel[i,2] - 1
    if class_counter[class_id] < retrain_sample_per_class:
        x_retrain[k] = trainData[i]
        y_retrain[k] = trainlabel[i, 2]
        k += 1
        class_counter[class_id] += 1
    elif class_counter[class_id] < retrain_sample_per_class + pretrain_test_sample_per_class:
        x_test_pretrain[l] = trainData[i]
        y_test_pretrain[l] = trainlabel[i, 0:2]
        l += 1
        class_counter[class_id] += 1
    else:
        x_pretrain[j] = trainData[i]
        y_pretrain[j] = trainlabel[i, 0:2]
        j += 1
        class_counter[class_id] += 1

# pdb.set_trace()

y_pretrain = y_pretrain.astype(int)
y_test_pretrain = y_test_pretrain.astype(int)
y_retrain = y_retrain.astype(int)


Y_pretrain1 = np.zeros((pretrain_size, 5))
Y_pretrain1[np.arange(pretrain_size), y_pretrain[:, 0]-1] = 1
Y_pretrain2 = np.zeros((pretrain_size, 4))
Y_pretrain2[np.arange(pretrain_size), y_pretrain[:, 1]-1] = 1

Y_test_pretrain1 = np.zeros((pretrain_test_size, 5))
Y_test_pretrain1[np.arange(pretrain_test_size), y_test_pretrain[:, 0]-1] = 1
Y_test_pretrain2 = np.zeros((pretrain_test_size, 4))
Y_test_pretrain2[np.arange(pretrain_test_size), y_test_pretrain[:, 1]-1] = 1

Y_retrain = np.zeros((retrain_size, 5))
Y_retrain[np.arange(retrain_size), y_retrain[:]-1] = 1

retrain_test_size = testLabel.shape[0]
Y_test_retrain = np.zeros((retrain_test_size,5))
Y_test_retrain[np.arange(retrain_test_size), testLabel[:,2]-1] = 1

pretrainData = x_pretrain.reshape((x_pretrain.shape[0], timestep, 2))
pretrainTestData = x_test_pretrain.reshape((x_test_pretrain.shape[0], timestep, 2))
retrainData = x_retrain.reshape((x_retrain.shape[0], timestep, 2))
retrainTestData = testData.reshape((testData.shape[0], timestep, 2))

def base_model():

    model_input = Input(shape=(timestep, 2))

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

    x1 = Dense(256)(x)
    x1 = Activation('relu')(x1)
    x1 = Dense(256, name='feature_extractor')(x1)
    x1 = Activation('relu')(x1)
    output1 = Dense(5, activation='softmax', name='Bandwidth')(x1)
    output2 = Dense(4, activation='softmax', name='Duration')(x1)

    model = Model(inputs=model_input, outputs=[output1, output2])
    opt = Adam(clipnorm = 1.)
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=opt, metrics=['accuracy'])

    return model

model = base_model()

model.fit(pretrainData, [Y_pretrain1, Y_pretrain2],
          validation_data = (pretrainTestData, [Y_test_pretrain1, Y_test_pretrain2]),
          batch_size = 64, epochs = 20, verbose = True, shuffle = True)

def target_model(source_model):
    feature_extractor_layer = source_model.get_layer('feature_extractor').output
    output = Dense(5, activation='softmax', name='TrafficClass')(feature_extractor_layer)

    target_model = Model(inputs=source_model.input, outputs=output)
    opt = Adam(clipnorm = 1.)
    target_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return target_model

new_model = target_model(model)

for layer in new_model.layers[:10]:
    layer.trainable = False
for layer in new_model.layers[10:]:
    layer.trainable = True

new_model.fit(retrainData, Y_retrain, validation_data = (retrainTestData, Y_test_retrain),
          batch_size=64, epochs=20, verbose=True, shuffle=True)
