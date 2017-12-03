import csv
import datetime
import os
# write file supporter
import numpy as np
import matplotlib.pyplot as plt


# results from serial version of matrix multiplication with size of matrices: 10, 100, 1000, 10000, 20000, 30000, 40000, 50000, ...

        
        


def write_data(values, file_name):
    myFile = open(file_name, 'a')
    myField = ['Episodes', 'Execution time', 'Win', 'Lost', 'Draw']
    data = dict(zip(myField, values))
    with myFile:
        writer = csv.DictWriter(myFile, fieldnames=myField)
        writer.writerow(data)
    myFile.close()
def get_name_result(board_size):
    name = str(os.getcwd()).replace('experiments','result/') + 'result'+'_'+ str(board_size)+'_'+ str(datetime.datetime.now()) +'.csv'
    data= []
    data.extend(['Episodes', 'Execution time', 'Win', 'Lost', 'Draw'])
    write_data(data,name)
    return name
def draw_graph(file_name):
    myFile = open(file_name, 'a')
    with open(file_name) as file:
        line = file.read()
        print(line)
    file.closed

def main():
    # serial_version = [0.004, 0.055, 10.185, 720.354, 1284.211]
    # matrix_sizes = ('10', '100', '1000', '10000', '20000')
    # x_pos = np.arange(len(serial_version))

    # plt.plot(serial_version, label = 'Serial Version')
    # # plt.plot(pthread_version, label = 'Pthread Version')

    # plt.xlabel('Matrix sizes (N)')
    # plt.xticks(x_pos, matrix_sizes)
    # plt.ylabel('Execution time (s)')
    # plt.title('Matrix Multiplication in Parallel Computing')
    # plt.grid(True)
    # plt.legend()

    # plt.show()
    file_name = '/home/antchil/Documents/btl/Kaithy_Reboot/result/result_5_2017-12-02 21:50:10.333220.csv'
    draw_graph(file_name)

if __name__ == "__main__":
    main()
