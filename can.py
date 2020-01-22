import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy import stats

class CAN:
    def __init__(self, size, tau, dt, kappa, beta, ws_param, h, plot_weights=True):
        self.size = size
        #self.u = np.random.random(size)

        self.u = np.zeros(size)
        self.u[5] = 1 #Initial active gridcell
        self.u_log = []

        self.grid_position_factor = 0.1 # mapped distance between two cells in meters
        self.grid_position_log = []
        self.real_position = (size - 5 - 1) * self.grid_position_factor

        self.tau = tau
        self.dt = dt
        self.beta = beta
        self.ws_param = ws_param
        self.h = h
        self.von_mises = True

        self.timestep_counter = 1
        self.max_speed = 0.08 # in m/s
        self.current_speed = 0.04
        self.movement = 0.5
        self.movement_r = 0.5
        self.movement_l = 0

        self.slope_calc_done = False
        self.line_reg_slope = 0
        self.cell_reg_slope = 0

        #Create weight matrix
        self.w = np.empty((size, size))
        self.w_right = np.empty((size,size))
        self.w_left = np.empty((size,size))
        phases = np.arange(-np.pi, np.pi, 2*np.pi/size)
        weight_shift = np.pi/self.ws_param

        #Connectivity values used in for weight matrix, Sanarmirskaya and Schöner - 2010 
        sigma = 1.0 #range
        c_exc = 1.0 #excitatory connectivity strength
        c_inh = 0.5 #inhibitory connectivity strength

        #Weight matrix using von Mises distribution
        if self.von_mises:
            for input_num, input_phase in enumerate(phases):
                self.w[:, input_num] = np.exp(kappa * np.cos(phases - input_phase))/np.exp(kappa)
                self.w_right[:, input_num] = np.exp(kappa * np.cos(phases + weight_shift - input_phase))/np.exp(kappa)
                self.w_left[:, input_num] = np.exp(kappa * np.cos(phases - weight_shift - input_phase))/np.exp(kappa)
        else:   
            for input_num, input_phase in enumerate(phases):
                self.w[:, input_num] = c_exc * np.exp( - pow((phases-input_phase),2) / (2* pow(sigma, 2))) - c_inh
                self.w_right[:, input_num] = c_exc * np.exp( - pow((phases + weight_shift - input_phase), 2) / (2 * pow(sigma, 2))) - c_inh
                self.w_left[:, input_num] = c_exc * np.exp( - pow((phases - weight_shift - input_phase), 2) / (2 * pow(sigma, 2))) - c_inh
        
        if False:
        #if plot_weights:
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
    def update_position(self,step_num):
        #TODO update speed constantly
##        if self.timestep_counter % 10 == 0:
##            alpha = 0.95
##            target_speed = random.random() * self.max_speed
##            self.current_speed = alpha * self.current_speed + (1-alpha) * target_speed
##            self.timestep_counter += 1
##        self.timestep_counter += 1
        
        #Directional speed values r (right) l(left)
        #if step_num < 1000:
        r = self.current_speed 
        l = 0
        self.real_position -= r * self.dt
##      else:
##          r = 0
##          l = self.current_speed
##          self.real_position += l * self.dt

        #Out of bounds control
        if self.real_position < 0:
            
            #Slope calculations for real position and activity position
            if self.slope_calc_done == False:
                self.slope_calc_done = True
                x = []
                y = []
                for pair in self.grid_position_log:
                    x.append(pair[0])
                    y.append(pair[1])
                line_reg = stats.linregress(x,y)
                #print(line_reg.slope)
                self.line_reg_slope = line_reg.slope
                active_cell_pos_a = self.size - 1 - np.argmax(self.u_log[0]) #TODO add OOB check
                active_cell_pos_b = self.size - 1 - np.argmax(self.u_log[step_num])
                a = np.array([[0,step_num*self.dt],[active_cell_pos_a,active_cell_pos_b]])
                cell_reg = stats.linregress(a)
                #print(cell_reg.slope)
                self.cell_reg_slope = cell_reg.slope
                
            self.grid_position_log.append((step_num*self.dt,math.nan))
            self.real_position = (self.size - 1) * self.grid_position_factor
            
        if self.real_position > (self.size - 1) * self.grid_position_factor:

            self.grid_position_log.append((step_num*self.dt,math.nan))
            self.real_position = 0
            
        dir_speed = [r,l]
        return dir_speed

    #Converts real world speed into useable vector form
    def translate_real_position(self,dir_speed):
        if dir_speed[1] == 0:
            self.movement_r = dir_speed[0]/self.max_speed
            self.movement = 1 - self.movement_r
            self.movement_l = 0
        else:
            self.movement_l = dir_speed[1]/self.max_speed
            self.movement = 1 - self.movement_l
            self.movement_r = 0
        
    
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
            p = self.update_position(step_num)
            self.translate_real_position(p)
            self.grid_position_log.append((step_num*self.dt,self.real_position/self.grid_position_factor))
        return [self.line_reg_slope,self.cell_reg_slope]


    def plot_activities(self):
        fig, ax = plt.subplots()
        self.u_log = np.array(self.u_log)
        mat = ax.matshow(self.u_log.transpose(), aspect="auto",
                         extent=(-self.dt/2, self.u_log.shape[0]*self.dt - self.dt/2, -0.5, self.u_log.shape[1]-0.5))
        ax.xaxis.set_ticks_position('bottom')
        ax.set_title("Network activities")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Unit number")
        x_val = [x[0] for x in self.grid_position_log]
        y_val = [x[1] for x in self.grid_position_log]
        plt.plot(x_val,y_val,'r-')
        cbaxes = fig.add_axes([0.92, 0.1, 0.01, 0.8])
        plt.colorbar(mat, cax=cbaxes)
        ax2 = ax.twinx()
        ax2.set_ylim(0,self.size * self.grid_position_factor)
        ax2.set_ylabel("Distance (m)")
        plt.subplots_adjust(right=0.8)
        plt.show()

    
