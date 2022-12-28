import csv
from io import FileIO


class Stack:
    def __init__(self):
        self.stack = []

    def peek(self):
        if self.isEmpty() == True:
            return None
        return self.stack[-1]

    def push(self, x):
        self.stack.append(x)

    def pop(self):
        return self.stack.pop(-1)

    def isEmpty(self):
        if len(self.stack) == 0:
            return True
        else:
            return False

    def getLen(self):
        return len(self.stack)

    def getList(self):
        return self.stack


def readSetting(settingName):
    # setting name, setting
    with open('settings.csv', 'r')as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            if row[0] == settingName:
                return row[1]
    return None


def writeSetting(settingName, val):
    with open('settings.csv', 'r')as f:
        csv_reader = csv.reader(f, delimiter=',')
        l = list(csv_reader)
    with open('settings.csv', 'w')as f:
        for i, row in enumerate(l):
            if len(row) == 0:
                continue
            if row[0] == settingName:
                l[i][1] = val
        writer = csv.writer(f, quotechar='"', lineterminator='\n')
        writer.writerows(l)
