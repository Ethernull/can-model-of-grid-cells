from can import CAN
from target import TARGET
import numpy as np
import matplotlib.pyplot as plt
import os.path
import math

peak_cell_num = 15
speed = 0.2

tg = TARGET(speed=speed,max_speed=0.8,size=20,peak_cell_num=peak_cell_num)

def single_run():
    #can = CAN(target=tg,size=20,tau=0.041, dt=0.01,kappa=0.1,beta =-8,ws_param=0.06, h=0)
    can = CAN(target=tg,size=20,tau=0.1, dt=0.01,kappa=0.1,beta =-8,ws_param=0.15, h=0)
    #can = CAN(target=tg,size=20,tau=0.06, dt=0.01,kappa=0.1,beta =-8,ws_param=0.15, h=0)
    can.init_network_activity(peak_cell_num=peak_cell_num,init_time=1)
    can.run(sim_time=20)
    can.plot_activities(u_out=True)
    print(can.slope_accuracy(speed,20,peak_cell_num))
    can.plot_single_cell(speed,20,0)
    plt.show()

def multiple_runs(overwrite_file):
    x = np.arange(0.01,0.7,0.01) #Value range for tau
    y = np.arange(0.05,1,0.05) #Value range for ws_param
    
    if os.path.isfile(str(speed) + '-Log.csv') == False or overwrite_file == True:
        print("No existing data file found;\nCalculating new data set..")
        mat_vals = np.empty((len(y),len(x)))
        progress = len(y)*len(x)/100
        counter = 0
        
        for j in range(0,len(x)):
            for i in range(0,len(y)):
                can = CAN(target=tg,size=20,tau=x[j], dt=0.01,kappa=0.1,beta =-8,ws_param=y[i], h=0)
                can.init_network_activity(peak_cell_num=peak_cell_num,init_time=1)
                can.run(sim_time=20)
                result = can.slope_accuracy(speed,20,peak_cell_num)
                mat_vals[i,j] = result
                counter += 1
                print(counter/progress,"%")
                
        np.savetxt(str(speed) + '-Log.csv',mat_vals,delimiter=',')

    else:
        print("Loading from existing data file..")
        mat_vals = np.loadtxt(str(speed) + '-Log.csv',delimiter=',')
        
    #print(mat_vals)
    mat_vals = np.clip(mat_vals, -100, 100)
    fig, ax = plt.subplots()
    mat = ax.matshow(mat_vals, origin='lower', aspect='auto', extent=(0.01-0.01/2, 0.69+0.01/2, 0.05-0.05/2, 0.95+0.05/2))
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title("Error of calculated slope to expected slope for "+str(speed)+" m/s")
    ax.set_xlabel("Tau [s]")
    ax.set_ylabel("Pi factor")
    ax.grid()
    cb = plt.colorbar(mat, ax=ax)
    cb.set_label("Error value")
    print(x)
    print(y)
    plt.show()

def multi_plot():
    matrices = []
    speeds = [0.2,0.4,0.6,0.8]
    for s in speeds:
        matrices.append(np.clip(np.loadtxt(str(s) + '-Log.csv',delimiter=','), -100, 100))
    fig, ax = plt.subplots(1, 4, sharey='row',figsize=(12,4))
    ax[0].set_ylabel("Pi factor")
    #fig.suptitle("Error of calculated slope to expected slope")
    for x in range(4):
        mat = ax[x].matshow(matrices[x],origin='lower', aspect='auto', extent=(0.01-0.01/2, 0.69+0.01/2, 0.05-0.05/2, 0.95+0.05/2))
        ax[x].set_title('v = '+str(speeds[x])+' m/s')
        ax[x].xaxis.set_ticks_position('bottom')
        ax[x].set_xlabel("Tau [s]")
        
    fig.autofmt_xdate()
    cb = plt.colorbar(mat, ax=ax)
    cb.set_label("Error of calculated slope to expected slope")
    #plt.show()

def average_matrix():
    matrices = []
    speeds = [0.2,0.4,0.6,0.8]
    for s in speeds:
        matrices.append(np.clip(np.loadtxt(str(s) + '-Log.csv',delimiter=','), -100, 100))
    matrix_sub_sum_a = np.add(matrices[0],matrices[1])
    matrix_sub_sum_b = np.add(matrices[2],matrices[3])
    matrix_sum = np.add(matrix_sub_sum_a,matrix_sub_sum_b) / 4
    fig, ax = plt.subplots()
    mat = ax.matshow(matrix_sum, origin='lower', aspect='auto', extent=(0.01-0.01/2, 0.69+0.01/2, 0.05-0.05/2, 0.95+0.05/2))
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title("Average Error of calculated slope to expected slope")
    ax.set_xlabel("Tau [s]")
    ax.set_ylabel("Pi factor")
    ax.grid()
    cb = plt.colorbar(mat, ax=ax)
    cb.set_label("Error value")
    plt.show()

def error_over_speed(tau,ws):
    speeds = np.arange(0.1,0.9,0.1)
    y = []
    for s in speeds:
        tg_var = TARGET(speed=s,max_speed=0.8,size=20,peak_cell_num=peak_cell_num)
        can = CAN(target=tg_var,size=20,tau=tau, dt=0.01,kappa=0.1,beta =-8,ws_param=ws, h=0)
        can.init_network_activity(peak_cell_num=peak_cell_num,init_time=1)
        can.run(sim_time=20)
        y.append(can.slope_accuracy(s,20,peak_cell_num))
    fig, ax = plt.subplots()
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title("Error in relation to speed [Tau="+str(tau)+"][WS="+str(ws)+"]")
    ax.set_xlabel("Speed [m/s]")
    ax.set_ylabel("Error")
    plt.plot(speeds,y,'r-')
    plt.ylim(-20, 20)
    plt.show()
        
        
#multiple_runs(0)
#single_run()
#multi_plot()
#average_matrix()
error_over_speed(0.1,0.15)
