import matplotlib.pyplot as plt
import numpy as np
import pickle
import os.path
from target2D import TARGET2D
from can2D import CAN2D

period_length = 0.5

#TARGET
speed = 0.6
max_speed = 0.8
direction = 0

tg2d = TARGET2D(speed,max_speed,direction,period_length)

#CAN
cells_per_row=10
cell_rows=9
init_activity=0

beta=-8
#ws_param = 1.75
#tau=0.1
tau = 0.11
ws_param = 1.55
dt=tau/10
h=0

def single_run():
    can2d = CAN2D(tg2d,cells_per_row,cell_rows,init_activity,dt,beta,ws_param,tau,h,period_length)

    can2d.init_network(1)
    can2d.run(sim_time=10)

    #can2d.plot_activities(u_out=True)
    #can2d.plot_single_cell(speed=0.2,sim_time=50,index=50)

    can2d.error_benchmark(print_details=True)
    can2d.plot_activities_interactive()
    

def multiple_runs(overwrite_file):
    x = np.arange(0.01,0.21,0.01) #Value range for tau
    y = np.arange(1,2,0.05) #Value range for ws_param
    delta_t = x/10
    
    if os.path.isfile(str(speed)+'ms_'+str(direction)+'degrees'+ '_log.pickle') == False or overwrite_file == True:
        if not overwrite_file:
            print("No existing data file found;\nCalculating new data set..")
        else:
            print("Overwriting existing data")
        mat_vals = np.empty((len(y),len(x)))
        progress = len(y)*len(x)/100
        counter = 0
        
        for j in range(0,len(x)):
            for i in range(0,len(y)):
                tg_2d = TARGET2D(speed,max_speed,direction,period_length)
                can2d = CAN2D(tg_2d,cells_per_row,cell_rows,init_activity,delta_t[j],beta,y[i],x[j],h,period_length)
                can2d.init_network(1)
                can2d.run(sim_time=5)
                result = can2d.error_benchmark(print_details=False)
                mat_vals[i,j] = result
                counter += 1
                print(counter/progress,"%")

        pickle.dump(mat_vals, open(str(speed)+'ms_'+str(direction)+'degrees'+ '_log.pickle',"wb"))   
        #np.savetxt(str(speed) + '-Log.csv',mat_vals,delimiter=',')

    else:
        print("Loading from existing data file..")
        #mat_vals = np.loadtxt(str(speed) + '-Log.csv',delimiter=',')
        mat_vals = pickle.load(open(str(speed)+'ms_'+str(direction)+'degrees'+ '_log.pickle',"rb"))

    mat_vals = np.clip(mat_vals, -100, 100)
    fig, ax = plt.subplots()
    mat = ax.matshow(mat_vals, origin='lower',cmap='seismic', aspect='auto', extent=(0.01-0.01/2, 0.20+0.01/2, 1.0-0.1/2, 1.95+0.01/2))
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title("Error of vector magnitudes for "+str(speed)+" m/s "+str(direction)+"°")
    ax.set_xlabel("Tau [s]")
    ax.set_ylabel("Shift factor")
    ax.set_facecolor('tab:gray')
    ax.grid()
    cb = plt.colorbar(mat, ax=ax)
    cb.set_label("Error value")
    mat.set_clim(-100,100)
    plt.show()
    #TODO plot error over speed/angle
    
def error_over_speed(tau,ws):
    speeds = np.arange(0.1,0.9,0.1)
    y = []
    for s in speeds:
        tg_var = TARGET2D(s,max_speed,direction,period_length)
        can2d = CAN2D(tg_var,cells_per_row,cell_rows,init_activity,dt,beta,ws_param,tau,h,period_length)
        can2d.init_network(1)
        can2d.run(sim_time=10)
        y.append(can2d.error_benchmark(print_details=False))
    fig, ax = plt.subplots()
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title("Error in relation to speed [Tau="+str(tau)+"][WS="+str(ws)+"]")
    ax.set_xlabel("Speed [m/s]")
    ax.set_ylabel("Error")
    plt.plot(speeds,y,'r-')
    plt.ylim(-20, 20)
    plt.show()

single_run()
#multiple_runs(overwrite_file = False)
#error_over_speed(tau,ws_param)
