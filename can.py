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

        #TODO replace shifting layers
        self.u_shift = []
        self.u_shift.append(self.u.copy())  #still layer
        self.u_shift.append(self.u.copy())  #right layer (placeholder)
        self.u_shift.append(self.u.copy())  #left  layer (placeholder)

        #Create weight matrix
        self.w = np.empty((size, size))
        phases = np.arange(-np.pi, np.pi, 2*np.pi/size)

        #Connectivity values used in for weight matrix, Sanarmirskaya and Schöner - 2010 
        sigma = 1.0 #range
        c_exc = 1.0 #excitatory connectivity strength
        c_inh = 0.5 #inhibitory connectivity strength

        #Weight matrix using von Mises distribution
        if von_mises:
            for input_num, input_phase in enumerate(phases):
                self.w[:, input_num] = np.exp(kappa * np.cos(phases - input_phase))/np.exp(kappa)
        else:   
            for input_num, input_phase in enumerate(phases):
                self.w[:,input_num] = c_exc * np.exp( - pow((phases-input_phase),2) / (2* pow(sigma,2))) - c_inh

        #Create multiple weight matrices based on needed shifting directions
        self.w_right = np.roll(self.w.copy(),10,axis=1)
        self.w_left = np.roll(self.w.copy(),10,axis=1)
        
        if plot_weights:
            fig, ax = plt.subplots()
            mat = ax.matshow(self.w)
            #mat = ax.matshow(self.w_right)
            ax.xaxis.set_ticks_position('bottom')
            ax.set_title("Weight matrix")
            ax.set_xlabel("Input unit number")
            ax.set_ylabel("Output unit number")
            plt.colorbar(mat, ax=ax)

    #---Consideration: Create new object class---
    def no_movement(self):
        dir_weights = []
        dir_weights.append(1.0)
        dir_weights.append(0.0)
        dir_weights.append(0.0)
        return dir_weights

    def moving_right(self):
        dir_weights = []
        dir_weights.append(0.0)
        dir_weights.append(1.0)
        dir_weights.append(0.0)
        return dir_weights
    
    #def moving_right(self):
    #    dir_weights = []
    #    dir_weights.append(random.randrange(0,10,1)/10)
    #    dir_weights.append(1.0 - dir_weights[0])
    #    dir_weights.append(0.0)
    #    return dir_weights
    
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
        #threshold = 0.5
        
        for step_num in range(int(round(sim_time/self.dt)) + 1):
            
            u_out = 1/(1 + np.exp(self.beta*(self.u - 0.5)))    #Activation function [Beta = -8](? = 0.5)
            exc_input = np.dot(u_out, self.w)                   #Sum: f(u(x, t)ω(x − x')dx'

            #TODO use shifting layers based on movement_signal and gridcell input
            #if move_signal[1] * self_u > threshold
            #u_out_right = 1/(1 + np.exp(self.beta*(self.u_shift[1] - 0.5)))
            #exc_input = np.dot(u_out_right, self.w_right)
            
            if self.von_mises:
                inh_input = max(0, np.sum(u_out) - 1)               #Inhibition, needed for von Mises distribution
            else:
                inh_input = 0
            
            self.u += (-self.u + exc_input - inh_input - self.h)/self.tau*self.dt
            self.u_log.append(self.u.copy())
            
            #Update all shifting layers
            self.u_shift[0] = self.u.copy()
            #self.u_shift[1] = self.u
            #self.u_shift[2] = self.u

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

    
