from can import CAN
from target import TARGET
import numpy as np
import matplotlib.pyplot as plt
import os.path

peak_cell_num = 15
speed = 0.8

tg = TARGET(speed=speed,max_speed=0.8,size=20,peak_cell_num=peak_cell_num)

def single_run():
    #can = CAN(target=tg,size=20,tau=0.041, dt=0.01,kappa=0.1,beta =-8,ws_param=0.06, h=0)
    can = CAN(target=tg,size=20,tau=0.041, dt=0.01,kappa=0.1,beta =-8,ws_param=0.06, h=0)
    can.init_network_activity(peak_cell_num=peak_cell_num,init_time=1)
    can.run(sim_time=20)
    can.plot_activities(u_out=True)

    print(can.slope_accuracy(speed,20,peak_cell_num))
    can.plot_single_cell(speed,19,20)
    plt.show()

def multiple_runs(overwrite_file):
    x = np.arange(0.05,1,0.05) #Value range for ws_param
    y = np.arange(0.01,0.2,0.01) #Value range for tau

    if os.path.isfile(str(speed) + '-Log.csv') == False or overwrite_file == True:
        print("No existing data file found;\nCalculating new data set..")
        mat_vals = np.empty((len(x),len(y)))
        progress = len(x)*len(y)/100
        counter = 0
        
        for j in range(0,len(y)):
            for i in range(0,len(x)):
                can = CAN(target=tg,size=20,tau=y[j], dt=0.01,kappa=0.1,beta =-8,ws_param=x[i], h=0)
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
    mat = ax.matshow(mat_vals, origin='lower', aspect='auto', extent=(y[0]-(y[1]-y[0])/2, y[-1]+(y[1]-y[0])/2, x[0]-(x[1]-x[0])/2, x[-1]+(x[1]-x[0])/2))
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title("Accuracy of grid slope to real slope in % for "+str(speed)+" m/s")
    ax.set_xlabel("Tau")
    ax.set_ylabel("Pi factor [weight shift]")
    #plt.axis([2,0.125,3.75,0.25])
    plt.colorbar(mat, ax=ax)
    #x = np.insert(x,0,0)
    #y = np.insert(y,0,0)
    #plt.gca().set_yticklabels(x)
    #plt.gca().set_xticklabels(y)
    print(x)
    print(y)
    plt.show()

def multi_plot():
    mat_vals1 = np.loadtxt(str(0.4) + '-Log.csv',delimiter=',')
    mat_vals2 = np.loadtxt(str(0.6) + '-Log.csv',delimiter=',')
    mat_vals3 = np.loadtxt(str(0.8) + '-Log.csv',delimiter=',')
    fig, ax = plt.subplots(1, 3, sharey='row')
    mat = ax[0].matshow(mat_vals1)
    mat = ax[1].matshow(mat_vals2)
    mat = ax[2].matshow(mat_vals3)
    # ax.xaxis.set_ticks_position('bottom')
    # ax.set_title("Accuracy of grid slope to real slope in % for "+str(speed)+" m/s")
    # ax.set_xlabel("Tau")
    # ax.set_ylabel("Pi factor [weight shift]")
    plt.colorbar(mat, ax=ax)
    plt.show()


multiple_runs(1)
# single_run()
# multi_plot()
