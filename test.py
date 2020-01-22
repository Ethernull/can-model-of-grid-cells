from can import CAN
import numpy as np
import matplotlib.pyplot as plt

#old: tau = 8, kappa = 4

single_test = False

if single_test:
    can = CAN(size=20, tau=3.5, dt=0.01, kappa=0.5, beta=-8,ws_param=4.5, h=0)
    result = can.run(sim_time=200)
    print(result[1]/result[0])
    can.plot_activities()


if single_test == False:
    x = np.arange(1,17,3.5) #Value range for tau
    y = np.arange(8,1,-1.5) #Value range for ws_param

    mat_vals = np.empty((len(x),len(y)))
    progress = len(x)*len(y)/100
    counter = 0
    
    for j in range(0,len(y)):
        for i in range(0,len(x)):
            can = CAN(size=20, tau=y[j], dt=0.01, kappa=0.5, beta=-8,ws_param=x[i], h=0)
            result = can.run(sim_time=200)
            mat_vals[i,j] = 100 * result[1]/result[0]
            counter += 1
            print(counter/progress,"%")

    print(mat_vals)
    fig, ax = plt.subplots()
    mat = ax.matshow(mat_vals)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title("Accuracy of grid slope to real slope in %")
    ax.set_xlabel("Tau")
    ax.set_ylabel("Fraction 1/x of pi [weight shift]")
    plt.colorbar(mat, ax=ax)
    x = np.insert(x,0,0)
    y = np.insert(y,0,0)
    plt.gca().set_xticklabels(y)
    plt.gca().set_yticklabels(x)
    plt.show()
