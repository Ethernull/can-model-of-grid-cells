import numpy as np
import matplotlib.pyplot as plt


class CAN:
    def __init__(self, size, tau, dt, kappa, beta, h,movement_mode, plot_weights=True, von_mises=True):
        self.u = np.random.random(size) 
        self.u_log = []
        self.tau = tau
        self.dt = dt
        self.beta = beta
        self.h = h
        self.von_mises = von_mises
        self.mode = movement_mode #0: none #1: right-only #2: left-only #3: random
        #TODO create shifting layers
        self.u_shift = []
        self.u_shift.append(self.u.copy())
        #TODO  append shifting layers for moving directions
        

        # create weight matrix
        self.w = np.empty((size, size))
        phases = np.arange(-np.pi, np.pi, 2*np.pi/size)

        #connectivity values used in Sanarmirskaya and Schöner - 2010 
        sigma = 3.0 #range : sigma (3.0 ; [3.0,3.0])
        c_exc = 0.9 #excitatory connectivity strength: c_exc (0.9 ; 0.6)
        c_inh = 5.0 #inhibitory connectivity strength: c_inh (5.0 ; 0.5)

        #TODO create multiple weight distributions based on needed shifting directions

        #Weight matrix using von Mises distribution
        if von_mises:
            for input_num, input_phase in enumerate(phases):
                self.w[:, input_num] = np.exp(kappa * np.cos(phases - input_phase))/np.exp(kappa)
        else:   
            for input_num, input_phase in enumerate(phases):
                self.w[:,input_num] = c_exc * np.exp( - pow((phases-input_phase),2) / (2* pow(sigma,2))) - c_inh
        
        if plot_weights:
            fig, ax = plt.subplots()
            mat = ax.matshow(self.w)
            ax.xaxis.set_ticks_position('bottom')
            ax.set_title("Weight matrix")
            ax.set_xlabel("Input unit number")
            ax.set_ylabel("Output unit number")
            plt.colorbar(mat, ax=ax)

    #---Consideration: Create new object class---
    def no_movement(self):
        return 0

    def moving_right(self):
        return 1
    
    def movement_signal(self):
        switcher = {
            0:self.no_movement,
            1:self.moving_right
            #2:moving left,
            #3:random_movement,
        }

        func = switcher.get(self.mode, lambda: "Invalid mode")
        return func()
    #------
    
    def run(self, sim_time):
        move_signal = self.movement_signal() 
        print(self.u)
        for step_num in range(int(round(sim_time/self.dt)) + 1):
            
            u_out = 1/(1 + np.exp(self.beta*(self.u_shift[move_signal] - 0.5)))    #Activation function [Beta = -8](? = 0.5)
            exc_input = np.dot(u_out, self.w)                   #Sum: f(u(x, t)ω(x − x')dx'
            if self.von_mises:
                inh_input = max(0, np.sum(u_out) - 1)               #Inhibition, needed for von Mises distribution
            else:
                inh_input = 0
            
            #TODO use shifting layers based on return value of movement_signal and gridcell input
            self.u += (-self.u + exc_input - inh_input - self.h)/self.tau*self.dt
            #self.u += (-self.u + exc_input - self.h)/self.tau*self.dt
            self.u_log.append(self.u.copy())
            self.u_shift[move_signal] = self.u #Update shifting layer

    def plot_activities(self):
        fig, ax = plt.subplots()
        self.u_log = np.array(self.u_log)
        mat = ax.matshow(self.u_log.transpose(), aspect="auto",
                         extent=(-self.dt/2, self.u_log.shape[0]*self.dt - self.dt/2, -0.5, self.u_log.shape[1]-0.5))
        ax.xaxis.set_ticks_position('bottom')
        ax.set_title("Network activities")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Unit number")
        plt.colorbar(mat, ax=ax)
        plt.show()

    
