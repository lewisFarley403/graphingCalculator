import csv

with open('settings.csv','r')as f:
    csv_reader = csv.reader(f, delimiter=',')
    l = list(csv_reader)
    print(l)
with open('settings.csv','w')as f:
    for i,row in enumerate(l):
        if len(row)==0:
            continue
        if row [0]=='angleMode':
            print('changing')
            l[i][1]=1
    writer = csv.writer(f,quotechar='"',lineterminator='\n')
    print(dir(writer))
    writer.writerows(l)
    