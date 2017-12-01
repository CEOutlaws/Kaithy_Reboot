import csv
import datetime
import os
# write file supporter
def get_name_result():
    return str(os.getcwd()).replace('experiments','result/') + 'result' +'_'+ str(datetime.datetime.now()) +'.csv'
    # def __init__(self):
    #     self._name = 'result' +'_'+ str(datetime.datetime.now()) +'.csv'

    # @property
    # def name(self):
    #     return self._name
        
        

def write_data(values, file_name ):
    myFile = open(file_name, 'a')
    myField = ['Episodes', 'Execution time', 'Win', 'Lost', 'Draw']
    data = dict(zip(myField, values))
    with myFile:
        writer = csv.DictWriter(myFile, fieldnames=myField)
        writer.writerow(data)
    myFile.close()

def main():
    print("hello")
    write_data(["1","haha"],'a.csv')

if __name__ == "__main__" :
    main()