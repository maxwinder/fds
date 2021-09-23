import os
import numpy as np
import pandas as pd

sessions = np.arange(25, 76)
data=[]

for session in sessions:
    directory = "./TXT/Session "+str(session)+" - "+str(1945+session)
    for filename in os.listdir(directory):
        f = open(os.path.join(directory, filename))
        if filename[0]==".": #ignore hidden files
            continue
        splt = filename.split("_")
        data.append([session, 1945+session, splt[0], f.read()])


df_speech = pd.DataFrame(data, columns=['Session','Year','ISO-alpha3 Code','Speech'])

df_speech.tail()

processedspeeches = []
for j in df_speech.Speech:
    processedspeeches.append(j.split(' ')[:100])

for i in range(len(processedspeeches)):
    for j in range(len(processedspeeches[i])):
        processedspeeches[i][j] = re.sub('\s+', '', processedspeeches[i][j])


processedspeeches[:5]
