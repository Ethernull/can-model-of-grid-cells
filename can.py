import numpy as np
import matplotlib.pyplot as plt
import random

class CAN:
    def __init__(self, size, tau, dt, kappa, beta, h,movement_mode, plot_weights=True, von_mises=True):
        self.size = size
        #self.u = np.random.random(size)

        self.u = np.zeros(size)
        self.u[5] = 1 #Initial active gridcell
        self.u_log = []

        self.position_factor = 0.1 # mapped distance between two cells in meters
        self.position_log = []
        self.real_position = (size - 5 - 1) * self.position_factor

        self.tau = tau
        self.dt = dt
        self.beta = beta
        self.h = h
        self.von_mises = von_mises

        #self.mode = movement_mode #0: none #1: right-only #2: left-only #3: random
        #self.movement = self.movement_signal()[0]
        #self.movement_r = self.movement_signal()[1]
        #self.movement_l = self.movement_signal()[2]

        #TODO fix speed variations
        self.movement = 0.5
        self.movement_r = 0.5
        self.movement_l = 0

        #Create weight matrix
        self.w = np.empty((size, size))
        self.w_right = np.empty((size,size))
        self.w_left = np.empty((size,size))
        phases = np.arange(-np.pi, np.pi, 2*np.pi/size)
        weight_shift = np.pi/20

        #Connectivity values used in for weight matrix, Sanarmirskaya and Schöner - 2010 
        sigma = 1.0 #range
        c_exc = 1.0 #excitatory connectivity strength
        c_inh = 0.5 #inhibitory connectivity strength

        #Weight matrix using von Mises distribution
        if von_mises:
            for input_num, input_phase in enumerate(phases):
                self.w[:, input_num] = np.exp(kappa * np.cos(phases - input_phase))/np.exp(kappa)
                self.w_right[:, input_num] = np.exp(kappa * np.cos(phases + weight_shift - input_phase))/np.exp(kappa)
                self.w_left[:, input_num] = np.exp(kappa * np.cos(phases - weight_shift - input_phase))/np.exp(kappa)
        else:   
            for input_num, input_phase in enumerate(phases):
                self.w[:, input_num] = c_exc * np.exp( - pow((phases-input_phase),2) / (2* pow(sigma, 2))) - c_inh
                self.w_right[:, input_num] = c_exc * np.exp( - pow((phases + weight_shift - input_phase), 2) / (2 * pow(sigma, 2))) - c_inh
                self.w_left[:, input_num] = c_exc * np.exp( - pow((phases - weight_shift - input_phase), 2) / (2 * pow(sigma, 2))) - c_inh
        
        if plot_weights:
            fig, ax = plt.subplots()
            mat = ax.matshow(self.w)
            #mat = ax.matshow(self.w_right)
            #mat = ax.matshow(self.w_left)
            ax.xaxis.set_ticks_position('bottom')
            ax.set_title("Weight matrix")
            ax.set_xlabel("Input unit number")
            ax.set_ylabel("Output unit number")
            plt.colorbar(mat, ax=ax)

    #Generates varying movement and returns current real position
    #TODO implement left direction
    def get_new_position(self): 
        max_speed = 0.08  #0.01 meters #TODO calculate max speed based on conversion factor
        r = random.random() * max_speed
        l = 0
        self.real_position -= r * self.dt
        if self.real_position < 0:
            self.real_position = (self.size - 1) * self.position_factor
        position = [r/max_speed,l/max_speed]
        return position

    #Calculates and updates movement values from real position
    def translate_real_position(self,position):
        self.movement_r = position[0] #/ self.position_factor
        self.movement = 1 - self.movement_r
        self.movement_l = 0 #TODO Implement opposite direction
        
    
    def run(self, sim_time):
        for step_num in range(int(round(sim_time/self.dt)) + 1):
            u_out = 1/(1 + np.exp(self.beta*(self.u - 0.5)))

            #Update all shifting layers: f(g-1+m)
            self.u_shift   = u_out * self.movement
            self.u_shift_r = u_out * self.movement_r
            self.u_shift_l = u_out * self.movement_l
            
            exc_input = np.dot(self.u_shift, self.w)
            exc_input_r = np.dot(self.u_shift_r,self.w_right)
            exc_input_l = np.dot(self.u_shift_l,self.w_left)
            
            if self.von_mises:
                inh_input = max(0, np.sum(u_out) - 1)               #Inhibition, needed for von Mises distribution
            else:
                inh_input = 0

            self.u += (-self.u + exc_input - inh_input - self.h + exc_input_r + exc_input_l)/self.tau*self.dt
            self.u_log.append(self.u.copy())
            p = self.get_new_position()
            self.translate_real_position(p)
            if step_num % 10 == 0:
                self.position_log.append(self.real_position/self.position_factor)


    def plot_activities(self):
        fig, ax = plt.subplots()
        self.u_log = np.array(self.u_log)
        mat = ax.matshow(self.u_log.transpose(), aspect="auto",
                         extent=(-self.dt/2, self.u_log.shape[0]*self.dt - self.dt/2, -0.5, self.u_log.shape[1]-0.5))
        ax.xaxis.set_ticks_position('bottom')
        ax.set_title("Network activities")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Unit number")
        plt.plot(self.position_log, 'r-')
        plt.colorbar(mat, ax=ax)
        plt.show()

    
