import numpy as np
import os
import glob
import csv
import re

i = 1
while i<6:
    wav_dir = './IEMOCAP/Session'+str(i)+'/sentences/wav'
    label_dir = './IEMOCAP/Session'+str(i)+'/dialog/EmoEvaluation'
    for sess in os.listdir(wav_dir):
        emotdir = label_dir+'/'+sess+'.txt'
        #emotfile = open(emotdir)
        #emot_map = {}
        with open(emotdir,'r') as emot_to_read:
            while True:
                line = emot_to_read.readline()
                if not line:
                    break
                if(line[0] == '['):
                    a=0
                    print(line)
                    t = line.split()
                    #print(t)
                    file_name = t[3]
                    print(file_name)
                    emotion = t[4]
                    print(emotion)
                    with open('label'+str(i)+'.csv','a',newline='') as csvfile: 
                        writer = csv.writer(csvfile)
                        #writer.writerow(["file","emotion"])
                        writer.writerow([[str(file_name)],[(emotion)]])
                # else:
                    # if(re.findall(r" hap ",line) or re.findall(r" sad ",line) or re.findall(r" neu ",line) or re.findall(r" ang ",line) or re.findall(r" sur ",line) or re.findall(r" fea ",line) or re.findall(r" dis ",line) or re.findall(r" fru ",line) or re.findall(r" exc ",line)):
                        # print(line)
    i=i+1


