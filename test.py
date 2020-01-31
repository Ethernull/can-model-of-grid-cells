from can import CAN
import numpy as np
import matplotlib.pyplot as plt
import os.path

#old: tau = 8, kappa = 4
#Note: kappa < 4 causes negative values for u

def single_test(move_speed):
    can = CAN(size=20, tau=0.75, dt=0.01, kappa=0.5, beta=-8,ws_param=4,speed=move_speed, h=0)
    result = can.run(sim_time=200)
    print(result[1]/result[0])
    can.plot_activities()
    can.plot_single_cell(index=0,sim_time=200)

def multiple_tests(move_speed,overwrite_file):
    x = np.arange(1,8,0.5) #Value range for ws_param
    y = np.arange(4,0,-0.25) #Value range for tau

    if os.path.isfile(str(move_speed) + '-Log.csv') == False or overwrite_file == True:
        print("No existing data file found;\nCalculating new data set..")
        mat_vals = np.empty((len(x),len(y)))
        progress = len(x)*len(y)/100
        counter = 0
        
        for j in range(0,len(y)):
            for i in range(0,len(x)):
                can = CAN(size=20, tau=y[j], dt=0.01, kappa=0.5, beta=-8,ws_param=x[i],speed=move_speed, h=0)
                result = can.run(sim_time=200)
                mat_vals[i,j] = 100 * result[1]/result[0]
                counter += 1
                print(counter/progress,"%")
                
        np.savetxt(str(move_speed) + '-Log.csv',mat_vals,delimiter=',')

    else:
        print("Loading from existing data file..")
        mat_vals = np.loadtxt(str(move_speed) + '-Log.csv',delimiter=',')
        
    #print(mat_vals)
    fig, ax = plt.subplots()
    mat = ax.matshow(mat_vals)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title("Accuracy of grid slope to real slope in % for "+str(move_speed)+" m/s")
    ax.set_xlabel("Tau")
    ax.set_ylabel("Fraction 1/x of pi [weight shift]")
    plt.colorbar(mat, ax=ax)
    #x = np.insert(x,0,0)
    #y = np.insert(y,0,0)
    #plt.gca().set_xticklabels(y)
    #plt.gca().set_yticklabels(x)
    print(x)
    print(y)
    plt.show()

single_test(0.01)
#multiple_tests(0.01,False)
