import csv
import datetime
import os
# write file supporter

        
        

def write_data(values, file_name ):
    myFile = open(file_name, 'a')
    myField = ['Episodes', 'Execution time', 'Win', 'Lost', 'Draw']
    data = dict(zip(myField, values))
    with myFile:
        writer = csv.DictWriter(myFile, fieldnames=myField)
        writer.writerow(data)
    myFile.close()
def get_name_result():
    name = str(os.getcwd()).replace('experiments','result/') + 'result' +'_'+ str(datetime.datetime.now()) +'.csv'
    data= []
    data.extend(['Episodes', 'Execution time', 'Win', 'Lost', 'Draw'])
    write_data(data,name)
    return name
def main():
    print("hello")
    write_data(["1","haha"],'a.csv')

if __name__ == "__main__" :
    main()