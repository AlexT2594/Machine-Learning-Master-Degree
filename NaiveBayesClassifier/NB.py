import re
import random

def trainWithFile(dataset):
    words = {}
    wordsProbHam = {}
    wordsProbSpam = {}
    documentHamCounter = 0
    documentSpamCounter = 0
    wordHamCounter = 0
    wordSpamCounter = 0

    
    with open(dataset,encoding='utf-8') as f:
        for line in f:
            lineWithClass = line.split('\t')
            c = lineWithClass[0]
            if(c == 'ham'):
                documentHamCounter += 1
            else:
                documentSpamCounter += 1
            phrase = lineWithClass[1]
            singleWords = re.split('\W+',phrase)
            for word in singleWords:
                word = word.lower()

                if(word in words):
                    
                    if(c == 'ham'):
                        
                        wordHamCounter += 1
                        words[word][0][1] = words[word][0][1] + 1
                    else:
                        wordSpamCounter += 1
                        words[word][1][1] = words[word][1][1] + 1
                    
                else:
                    words[word] = []
                    
                    if(c == 'ham'):
                        wordHamCounter += 1
                        words[word].append(['ham',1])
                        words[word].append(['spam',0])
                    else:
                        wordSpamCounter += 1
                        words[word].append(['ham',0])
                        words[word].append(['spam',1])


    pcHam = documentHamCounter / (documentHamCounter + documentSpamCounter)
    pcSpam = documentSpamCounter / (documentHamCounter + documentSpamCounter)

    
    for word in words:
        tfHam = words[word][0][1]
        wordsProbHam[word] = (tfHam +1) / (wordHamCounter + len(words.keys()))
        tfSpam = words[word][1][1]
        wordsProbSpam[word] = (tfSpam+1) / (wordSpamCounter + len(words.keys()))
    

    ans = []
    ans.extend((pcHam,pcSpam,wordsProbHam,wordsProbSpam))
    print(len(words.keys()))
    print(len(wordsProbHam.keys()))
    print(len(wordsProbSpam.keys()))
    print('Total ham words: ' + str(wordHamCounter))
    print('Total spam words: ' + str(wordSpamCounter))
    print('Total documents: ' + str(documentHamCounter + documentSpamCounter))

          

    print(words['_'])
    return ans

def trainWithList(dataset):
    words = {}
    wordsProbHam = {}
    wordsProbSpam = {}
    documentHamCounter = 0 #tj -> tHam
    documentSpamCounter = 0 #tj -> tSpam
    wordHamCounter = 0 #TFj -> total number of words in Ham
    wordSpamCounter = 0

    
    for line in dataset:
        lineWithClass = line.split('\t')
        c = lineWithClass[0]
        if(c == 'ham'):
            documentHamCounter += 1
        else:
            documentSpamCounter += 1
        phrase = lineWithClass[1]
        singleWords = re.split('\W+',phrase)
        for word in singleWords:
            word = word.lower()
            if(word in words):
                
                if(c == 'ham'):
                    wordHamCounter += 1
                    words[word][0][1] += 1
                else:
                    wordSpamCounter += 1
                    words[word][1][1] += 1
                
            else:
                words[word] = []
                
                if(c == 'ham'):
                    wordHamCounter += 1
                    words[word].append(['ham',1])
                    words[word].append(['spam',0])
                else:
                    wordSpamCounter += 1
                    words[word].append(['ham',0])
                    words[word].append(['spam',1])

    pcHam = documentHamCounter / (documentHamCounter + documentSpamCounter)
    pcSpam = documentSpamCounter / (documentHamCounter + documentSpamCounter)
    for word in words:
        tfHam = words[word][0][1]
        wordsProbHam[word] = (tfHam +1) / (wordHamCounter + len(words.keys()))
        tfSpam = words[word][1][1]
        wordsProbSpam[word] = (tfSpam+1) / (wordSpamCounter + len(words.keys()))
    
    ans = []
    ans.extend((pcHam,pcSpam,wordsProbHam,wordsProbSpam,words))
    return ans   

def classify(data,sms):
    pcHam = data[0]
    pcSpam = data[1]
    wordsProbHam = data[2]
    wordsProbSpam = data[3]
    words = data[4]
    
    messageWithClass = sms.strip().split('\t')
    message = re.split('\W+',messageWithClass[1])
    whatIs  = messageWithClass[0]

    pHam = 0
    pSpam = 0

    prodHam = 1
    prodSpam = 1
    for i in range(len(message)):
        
        singleWord = message[i].lower()
        if( singleWord not in words ):
            continue
        prodHam = prodHam * wordsProbHam[singleWord]
        prodSpam = prodSpam * wordsProbSpam[singleWord]
    pHam = pcHam * prodHam
    pSpam = pcSpam * prodSpam

    ans = []
    if pHam > pSpam:
        classification = 'ham'
    else:
        classification = 'spam'
    ans.extend((pHam,pSpam,classification,whatIs))
    return ans

def createKFold(dataset):
    kSets = []
    lines = []
    with open(dataset,encoding='utf-8') as f:
        for line in f:
            lines.append(line)
    for i in range(9):
        kSets.append([])
        for j in range(557):
            random_index = random.randrange(0,len(lines))
            line = lines[random_index]
            lines.pop(random_index)
                
            kSets[i].append(line)
    kSets.append([]) 
    kSets[9].extend(lines)
    return kSets

def createTrainingSetWithClassifySet(kSets):
    table = []
    for i in range(len(kSets)):
        classifySet = kSets[i]
        trainingSet = []
        for j in range(len(kSets)):
            if(i == j):
                continue
            trainingSet.extend(kSets[j])
        table.append([])
        table[i].append(classifySet)
        table[i].append(trainingSet)

    return table


def accuracy(dataset):
    accuracyList = []
    table = createTrainingSetWithClassifySet(createKFold(dataset))
    for i in range(len(table)):
        dataTrain = trainWithList(table[i][1])
        classifySet = table[i][0]
        accuracySingle = 0
        for j in range(len(classifySet)):
            classifyResult = classify(dataTrain,classifySet[j])
            classification = classifyResult[2]
            whatIs = classifyResult[3]
            if(classification == whatIs):
                accuracySingle += 1
        finalSingleAccuracy = accuracySingle / len(classifySet)
        accuracyList.append(finalSingleAccuracy)

    return accuracyList


data = trainWithFile('SMSSpamCollection')

accuracyList = accuracy('SMSSpamCollection')
accuracy = 0
for i in range(len(accuracyList)):
    accuracy += accuracyList[i]
    
print("Accuracy: " + str(accuracy/10))




            



    
