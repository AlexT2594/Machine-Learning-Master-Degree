import os
from random import shuffle
files = os.listdir("drebin/feature_vectors")
shuffle(files)

def filterFiles(files,howManyMalwares,howManyNotMalware):
    malwares = []
    filteredFiles = []
    counterMalware = 0
    counterNotMalware = 0
    with open("sha256_family.csv",encoding="utf-8") as malwareFile:
        for line in malwareFile:
            lineWithFamily = line.split(",")
            malware = lineWithFamily[0]
            if(malware == "sha256"):
                continue
            family = lineWithFamily[1].strip()
            malwares.append(malware)
    for file in files:
        if(file in malwares and counterMalware < howManyMalwares):
            filteredFiles.append(file)
            counterMalware += 1
        elif(file not in malware and counterNotMalware < howManyNotMalware):
            filteredFiles.append(file)
            counterNotMalware += 1
    shuffle(filteredFiles)
    return filteredFiles

def prepareData(dataDir,malwareData,dataFileNames):
    data = {}
    fileCounter = 0
    jumped = False
    for fileName in dataFileNames:
        jumped = False
 #       if(fileCounter == 4000):
#            break
        if(fileName == '.DS_Store' or os.stat(dataDir + "/" + fileName).st_size == 0):
            continue
        data[fileName] = []
        data[fileName].append({'url':"",'feature':"",'provider':"",'call':"",'activity':"",'intent':"",'api_call':"",'real_permission':"",'permission':"",'service_receiver':""}) # data[fileName][0] is a dictionary of attributes while data[fileName][1] is the class of the instance
        with open(dataDir + "/" + fileName,encoding="utf-8") as f:
            attributesForSingleFile = {'url':[],'feature':[],'provider':[],'call':[],'activity':[],'intent':[],'api_call':[],'real_permission':[],'permission':[],'service_receiver':[]}
            for line in f:
                if("::" not in line):
                    data.pop(fileName)
                    jumped = True
                    fileCounter += 1
                    break
                attributeLine = line.split("::")
                attribute = attributeLine[0]
                value = attributeLine[1].strip()
                
                attributesForSingleFile[attribute].append(value)
                counter = 0
                for attr,val in attributesForSingleFile.items():
                    stringToAddToFileAttributes = ""
                    counter = 0
                    length = len(val)
                    for elem in val:
                        stringToAddToFileAttributes += elem
                        if(counter < length - 1):
                            stringToAddToFileAttributes += "&"
                        counter += 1
                    data[fileName][0][attr] = stringToAddToFileAttributes
            if jumped:
                continue
            for attr,val in data[fileName][0].items():
                if(val == ""):
                    data[fileName][0][attr] = '?'
            fileCounter += 1
    malwares = []
    with open(malwareData,encoding="utf-8") as malwareFile:
        for line in malwareFile:
            lineWithFamily = line.split(",")
            malware = lineWithFamily[0]
            if(malware == "sha256"):
                continue
            family = lineWithFamily[1].strip()
            malwares.append(malware)
    for app in data.keys():
        if(app in malwares):
            data[app].append("malware")
        else:
            data[app].append("not-malware")
            
    return data

def writeAttributes(dataDir,dataFileNames):
    with open("malware.arff",mode="w",encoding="utf-8") as output:
        output.write("@relation malware\n")
        attributes = {'url':dict(),'feature':dict(),'provider':dict(),'call':dict(),'activity':dict(),'intent':dict(),'api_call':dict(),'real_permission':dict(),'permission':dict(),'service_receiver':dict()} ##these are the general attributes
        encoder = {'url':0,'feature':0,'provider':0,'call':0,'activity':0,'intent':0,'api_call':0,'real_permission':0,'permission':0,'service_receiver':0}
        lengths = {}
        fileCounter = 0
        for fileName in dataFileNames:
            jumped = False
 #           if(fileCounter == 4000):
#                break
            attributesForSingleFile = {'url':[],'feature':[],'provider':[],'call':[],'activity':[],'intent':[],'api_call':[],'real_permission':[],'permission':[],'service_receiver':[]}
            if(fileName == '.DS_Store' or os.stat(dataDir + "/" + fileName).st_size == 0):
                continue
            with open(dataDir + "/" + fileName,encoding="utf-8") as f:
                for line in f:
                    if("::" not in line):
                        jumped = True
                        fileCounter += 1
                        break
                    attributeLine = line.split("::")
                    attribute = attributeLine[0].strip()
                    value = attributeLine[1].strip()

                    attributesForSingleFile[attribute].append(value)         
            
            if jumped:
                continue
            for attr,val in attributesForSingleFile.items():
                stringToAddToGlobalAttributes = ""
                counter = 0
                length = len(val)
                for elem in val:
                    
                    stringToAddToGlobalAttributes += elem
                    if(counter < length - 1):
                        stringToAddToGlobalAttributes += "&"
                    counter += 1
                if(stringToAddToGlobalAttributes == ""):
                    continue

                if(stringToAddToGlobalAttributes not in attributes[attr]):
                    attributes[attr][stringToAddToGlobalAttributes] = encoder[attr]
                    encoder[attr] += 1
            fileCounter += 1
        counter = 0
        for attr,d in attributes.items():
            length = len(d)
            output.write("@attribute " + attr + " {")
            for elem,encoding in d.items():
                output.write(str(encoding))
                    
                if(counter < length - 1):
                    output.write(",")
                counter += 1
            output.write("}\n")
            counter = 0
        output.write("@attribute Class {malware,not-malware}\n@data\n")
    return attributes

def writeData(data,attributesWithEncoder):
    with open("malware.arff",mode="a",encoding="utf-8") as output:
        for app,val in data.items():
            attributes = val[0]
            classification = val[1]
            for attribute,value in attributes.items():
                if(value == "?"):
                    output.write("?")
                else:
                    output.write(str(attributesWithEncoder[attribute][value]))
                output.write(",")
            output.write(classification + "\n")

def writeAttributesAlt(dataDir,dataFileNames):
    fileCounter = 0
    with open("malwareAlt.arff",mode="w",encoding="utf-8") as output:
        output.write("@relation malware\n")
        attributes = {}
        counter = 0
        for fileName in dataFileNames:
 #           if(fileCounter == 4000):
#                break
            if(fileName == '.DS_Store' or os.stat(dataDir + "/" + fileName).st_size == 0):
                continue
            with open(dataDir + "/" + fileName,encoding="utf-8") as f:
                for line in f:
                    if("::" not in line):
                        break
                    attributeLine = line.split("::")
                    attribute = attributeLine[0].strip()
                    value = attributeLine[1].strip()

                    if(value not in attributes):
                        attributes[value] = counter
                        output.write("@attribute " + str(counter) + " {0,1}\n")
                        counter += 1
            fileCounter += 1
        output.write("@attribute Class {malware,not-malware}\n@data\n")

    return attributes

def writeDataAlt(dataDir,dataFileNames,malwareData,attributes):
    malwares = []
    fileCounter = 0
    with open(malwareData,encoding="utf-8") as malwareFile:
        for line in malwareFile:
            lineWithFamily = line.split(",")
            malware = lineWithFamily[0]
            if(malware == "sha256"):
                continue
            malwares.append(malware)
    
    with open("malwareAlt.arff",mode="a",encoding="utf-8") as output:
        for fileName in dataFileNames:
 #           if(fileCounter == 4000):
#                break
            jumped = False
            start = True
            indices = []
            if(fileName == '.DS_Store' or os.stat(dataDir + "/" + fileName).st_size == 0):
                continue
            with open(dataDir + "/" + fileName,encoding="utf-8") as f:
                for line in f:
                    if("::" not in line):
                        jumped = True
                        fileCounter += 1
                        break
                    if(start):
                        output.write("{")
                        start = False
                    attributeLine = line.split("::")
                    value = attributeLine[1].strip()
                    index = attributes[value]
                    if(index not in indices):
                        indices.append(index)
                if(jumped):
                    continue
                else:
                    indices.sort()
                    
                    for index in indices:
                        output.write(str(index) + " 1, ")
                    if(fileName in malwares):                       
                        output.write(str(len(attributes)) + " malware}\n")
                    else:
                        output.write(str(len(attributes)) + " not-malware}\n")
            fileCounter += 1

#files2 = filterFiles(files,150,2850)
filteredFiles = filterFiles(files,200,3800)
filteredFilesAlt = filterFiles(files,1200,2800)           
data = prepareData("drebin/feature_vectors","sha256_family.csv",files)
attributes = writeAttributes("drebin/feature_vectors",files)
writeData(data,attributes)

attributes = writeAttributesAlt("drebin/feature_vectors",files)
writeDataAlt("drebin/feature_vectors",files,"sha256_family.csv",attributes)


        


            
    


