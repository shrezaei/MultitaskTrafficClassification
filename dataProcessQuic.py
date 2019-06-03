import pdb
import os
import numpy as np

ComputeInterArrival = True
DescretesizeLength = False
DirectionLengthCombined = True
NormalizeLength = True
NormalizeInterArrival = True
MaxLength = 1434
MaxInterArrival = 1
Starting_point = 0
StartingPointMultiply = 13
Num_of_extracted_subflow = 100
PaddingEnable = True
PadAtTheBegining = True
PaddingThreshold = 20


CompureStatisticsInThisScript = True

NumOfCrossValidationFolds = 1

np.random.seed(10)

timestep = 120
SkipPacketsForSampling = 1
IncrementalSampling = False
NumberOfSamplesUntiIncrement = 10000
IncrementalStepMultiplier = 1


def loadData(dirPath, class_label,  extractedFlows = 0):
    #If it is not set, use the global value
    if extractedFlows == 0:
        extractedFlows = Num_of_extracted_subflow

    pathDir = os.listdir(dirPath)

    datalist = []
    labellist = []
    FileCounter = 0
    FlowCounter = 0
    SubflowCounter = 0

    # added by Shahbaz
    custom_features = [
        # 0,    #timestamp
        1,  # RelativeTime
        2,  # length
        # 3    #Direction
    ]

    for folder, subs, files in os.walk(dirPath):

        # if FlowCounter > 1000:
        #     break

        np.random.shuffle(files)

        for file in files:
            filename = folder + "/" + file
            with open(filename) as f:
                
                FileCounter += 1
                
                EntireFile = []
                for line in f:
                    data = line.split()
                    try:
                        EntireFile.append(data)
                    except:
                        print(EntireFile)
                        pdb.set_trace()
                try:
                    EntireFile = np.array(EntireFile).astype(np.float)
                except:
                    print(EntireFile)
                    pdb.set_trace()

                FileLenght = len(EntireFile)

                if CompureStatisticsInThisScript:
                    Duration = EntireFile[-1,1]
                    Bandwidth = np.sum(EntireFile[:,2])/Duration
                    temp_label = [Bandwidth, Duration, class_label]

                SubflowFromAFile = 0

                #Skip the fist few packets in the file
                if(Starting_point!=0):
                    for jjj in range(Starting_point):
                        line = f.readline()
                        if not line:
                            break

                for subflow in range(extractedFlows):

                    startingPoint = 0
                    if subflow == 0:
                        startingPoint = 0
                    else:
                        startingPoint = np.random.randint(1, FileLenght-timestep)
                    # startingPoint = Starting_point + subflow*StartingPointMultiply
                    
                    linedata = []
                    Prev_time = 0;  #Time of the first packet in the subflow
    
                    numOfSamples = 0
                    i = startingPoint
                    SkipSamples = SkipPacketsForSampling
                    while(numOfSamples < timestep):
                        if i>=FileLenght:
                            break

                        data = list(EntireFile[i])  #To clone the list, not refering to the same list
                        
                        #shahbaz: To descretesize the the length
                        if DescretesizeLength:
                            data[2] = str(int(int(data[2])/100))

                        if DirectionLengthCombined:
                            if data[3]=="0":
                                if float(data[2])>0:
                                    data[2] = str(-1 * float(data[2]))
                                    
                        if NormalizeLength:
                            data[2] = str(float(data[2])/MaxLength)
                            
                        if ComputeInterArrival:
                            if i==startingPoint:
                                Prev_time = float(data[1])
                                data[1] = str(0)
                            else:
                                temporary = str(float(data[1]) - Prev_time)
                                Prev_time = float(data[1])
                                data[1] = temporary
                        if NormalizeInterArrival:
                            ttt = float(data[1]) / MaxInterArrival
                            if ttt > 1:
                                ttt=1
                            data[1]=(ttt-0.5)*2
                            

                        try:
                            data2 = [float(data[j]) for j in custom_features]
                        except (IndexError, ValueError) as e:
                            pass
                            print("Couldn't retrieve all data",filename)
                        else:
                            linedata += data2
                
                        numOfSamples += 1
                        i += SkipSamples
                        if IncrementalSampling:
                            if numOfSamples % NumberOfSamplesUntiIncrement == 0:
                                SkipSamples = int(SkipSamples*IncrementalStepMultiplier)

    
                    if (len(linedata) < len(custom_features) * timestep):
                        if (PaddingThreshold > len(linedata)/len(custom_features) ):
                            continue
                        #print(linedata)
                        if (PaddingEnable):
                            while(len(linedata) < len(custom_features) * timestep):
                                pad = []
                                pad.extend(np.ones(len(custom_features)) * 0)
                                if PadAtTheBegining:
                                    pad.extend(linedata)
                                    linedata = pad
                                else:
                                    linedata.extend(pad)
                            #print(linedata)
                        else:
                            continue
                    np.nan_to_num(linedata)
                    datalist.append(linedata)

                    SubflowCounter+=1
                    SubflowFromAFile+=1

                total_labels = [temp_label] * SubflowFromAFile
                labellist.extend(total_labels)
                FlowCounter+=1
                print(filename,temp_label)
    ratio = SubflowCounter/FlowCounter
    print(dirPath + ":" + str(FlowCounter) + "/" + str(len(pathDir)) + " - Subflows:" + str(SubflowCounter) + " - Ratio:", str(ratio))
    return (np.array(datalist), np.array(labellist))


if __name__ == "__main__":

    BaseDirectory = "Data/pretraining"
    (data1, label1) = loadData(BaseDirectory + "/Google Drive", 1, extractedFlows=1)
    (data2, label2) = loadData(BaseDirectory + "/Youtube", 2,  extractedFlows=1)
    (data3, label3) = loadData(BaseDirectory + "/Google Doc", 3,  extractedFlows=1)
    (data4, label4) = loadData(BaseDirectory + "/Google Search", 4, extractedFlows=1)
    (data5, label5) = loadData(BaseDirectory + "/Google Music", 5, extractedFlows=1)

    test_size = 30
    val_size = 30
    train1 = data1[:-(test_size+val_size)]
    train2 = data2[:-(test_size+val_size)]
    train3 = data3[:-(test_size+val_size)]
    train4 = data4[:-(test_size+val_size)]
    train5 = data5[:-(test_size+val_size)]
    val1 = data1[-(test_size+val_size):-test_size]
    val2 = data2[-(test_size+val_size):-test_size]
    val3 = data3[-(test_size+val_size):-test_size]
    val4 = data4[-(test_size+val_size):-test_size]
    val5 = data5[-(test_size+val_size):-test_size]
    test1 = data1[-test_size:]
    test2 = data2[-test_size:]
    test3 = data3[-test_size:]
    test4 = data4[-test_size:]
    test5 = data5[-test_size:]

    trainL1 = label1[:-(test_size+val_size)]
    trainL2 = label2[:-(test_size+val_size)]
    trainL3 = label3[:-(test_size+val_size)]
    trainL4 = label4[:-(test_size+val_size)]
    trainL5 = label5[:-(test_size+val_size)]
    valL1 = label1[-(test_size+val_size):-test_size]
    valL2 = label2[-(test_size+val_size):-test_size]
    valL3 = label3[-(test_size+val_size):-test_size]
    valL4 = label4[-(test_size+val_size):-test_size]
    valL5 = label5[-(test_size+val_size):-test_size]
    testL1 = label1[-test_size:]
    testL2 = label2[-test_size:]
    testL3 = label3[-test_size:]
    testL4 = label4[-test_size:]
    testL5 = label5[-test_size:]

    train_data = np.concatenate((train1, train2, train3, train4, train5), axis=0)
    val_data = np.concatenate((val1, val2, val3, val4, val5), axis=0)
    test_data = np.concatenate((test1, test2, test3, test4, test5), axis=0)

    train_label = np.concatenate((trainL1, trainL2, trainL3, trainL4, trainL5), axis=0)
    val_label = np.concatenate((valL1, valL2, valL3, valL4, valL5), axis=0)
    test_label = np.concatenate((testL1, testL2, testL3, testL4, testL5), axis=0)

    np.save("trainData.npy", train_data)
    np.save("trainLabel.npy", train_label)

    np.save("valData.npy", val_data)
    np.save("valLabel.npy", val_label)

    np.save("testData.npy", test_data)
    np.save("testLabel.npy", test_label)

    print(train_data.shape, train_label.shape)
    print(val_data.shape, val_label.shape)
    print(test_data.shape, test_label.shape)
