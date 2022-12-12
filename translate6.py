import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE



def main():
    #Load raw data
    rawData = pd.read_csv("data.csv")

    #Separating X and Y data
    x, y = getXYdata(rawData)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y)
    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train))
    x_test = pd.DataFrame(scaler.fit_transform(x_test))

    sm = SMOTE(random_state = 2)
    x_train_res, y_train_res = sm.fit_resample(x_train, y_train)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_dim=20, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(20, activation='softmax'),
    ])

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics='accuracy')



def normalize(word):
    normalizedWord = np.zeros((4, 5))
    #Number of letters
    normalizedWord[1][0] = len(word)


    #Number of vowels
    vowel = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
    count = 0
    for letter in word:
        if letter in vowel:
            count += 1
    normalizedWord[1][1] = float(count)

    #Last letter
    normalizedWord[1][2] = ord(word[-1].lower())

    #First letter
    normalizedWord[1][2] = ord(word[0].lower())

    #Second letter
    normalizedWord[0][0] = ord(word[1].lower())

    #Second-last letter
    normalizedWord[0][1] = ord(word[-2].lower())

    #Letter E
    for letter in word:
        if letter == 'e' or letter == 'E':
            normalizedWord[0][2] += 1

        if letter == 't' or letter == 'T':
            normalizedWord[0][3] += 1

        if letter == 'o' or letter == 'O':
            normalizedWord[0][4] += 1


    #Same letters
    chars = "abcdefghijklmnopqrstuvwxyz"
    for k in range(26):
        count = word.count(chars[k])
        if count > 1:
            normalizedWord[1][4] += 1

    #Following latters
    for i in range(len(word)-1):
        #Latter D followed by vowel
        if word[i].lower() == 'd' and word[i+1] in vowel:
            normalizedWord[2][0] += 1

        #Latter B followed by vowel    
        if word[i].lower() == 'b' and word[i+1] in vowel:
            normalizedWord[2][1] += 1

        #Latter C followed by vowel
        if word[i].lower() == 'c' and word[i+1] in vowel:
            normalizedWord[2][2] += 1

        #Latter H followed by vowel
        if word[i].lower() == 'h' and word[i+1] in vowel:
            normalizedWord[2][3] += 1

        #Latter J followed by vowel
        if word[i].lower() == 'j' and word[i+1] in vowel:
            normalizedWord[2][4] += 1
    if len(word)>5:
        for i in range(5):
            #Few Letters
            normalizedWord[3][i] = ord(word[i].lower())
    else:
        for i in range(len(word)):
            normalizedWord[3][i] = ord(word[i].lower())

    return normalizedWord.flatten()

def getXYdata(rawData):
    rawx = rawData.drop('cro', axis=1)
    rawy = rawData.drop('foreign', axis=1)

    #Make array to store normalized data
    dataRaw = pd.read_csv("dataold.csv")
    data = dataRaw.to_numpy()
    indexCounter = 0
    indexChecker = True
    correctedData = np.zeros((1742, 2), dtype=object)
    for line in data:
        indexChecker = True
        croWord = ""
        for word in line:
            if indexChecker==True:
                croWord = word
                indexChecker = False
            correctedData[indexCounter][0] = croWord
            correctedData[indexCounter][1] = word
            indexCounter += 1

    npdatay = np.array(correctedData[:, 0])
    pddatay = pd.DataFrame(npdatay, columns=["cro"]) 

    #Making 2 column data
    trainData = np.zeros((1742, 2, 20))
    for i in range(1742):
        for j in range(2):
            if correctedData[i][j] != 0:
                trainData[i][j] = normalize(correctedData[i][j])

    #Separating train data
    dataLabels = trainData[:, 0]
    dataFeatures = trainData[:, 1]


    npdatax = np.array(dataFeatures)

    pddatax = pd.DataFrame(npdatax, columns=["a", "b", "c","d","e","f","g","h","i","j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "z"])

    return pddatax, pddatay


if __name__=="__main__":
    main()