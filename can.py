import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
import random
import math
from scipy import stats
from scipy.signal import find_peaks
from target import TARGET

class CAN:
    def __init__(self,target, size, tau, dt, kappa, beta, ws_param, h, plot_weights=True,period_length=0.5):
        self.target = target
        self.size = size
        self.u = np.zeros(self.size)

        self.u_out_log = []             #Log for grid cell values after calculations
        self.u_log = []                 #Log for grid cell values before calculations, after applying sigmoid

        self.cell_distance = period_length/size  #Mapped distance between two cells in meters

        self.init_time = 0
        self.spatial_bin = np.zeros(int(size*self.cell_distance*100))
        self.activity_sums = np.zeros(int(size*self.cell_distance*100))

        self.tau = tau              #Time constant
        self.dt = dt                #Delta time
        self.beta = beta            #Sigmoid factor
        self.ws_param = ws_param    #Weight shift parameter
        self.h = h                  #Negative resting level
        self.von_mises = True
        
        #Create weight matrix
        self.w = np.empty((size, size))
        self.w_right = np.empty((size,size))
        self.w_left = np.empty((size,size))
        phases = np.arange(-np.pi, np.pi, 2*np.pi/size)
        weight_shift = 2* np.pi * self.ws_param
        weight_strength = 1

        #Connectivity values used in for weight matrix, Sanarmirskaya and Schöner - 2010 
        sigma = 1.0 #range
        c_exc = 1.0 #excitatory connectivity strength
        c_inh = 0.5 #inhibitory connectivity strength

        #Weight matrix using von Mises distribution
        if self.von_mises:
            for input_num, input_phase in enumerate(phases):
                self.w[:, input_num] = weight_strength * np.exp(kappa * np.cos(phases - input_phase))/np.exp(kappa)
                self.w_right[:, input_num] = weight_strength * np.exp(kappa * np.cos(phases + weight_shift - input_phase))/np.exp(kappa)
                self.w_left[:, input_num] = weight_strength * np.exp(kappa * np.cos(phases - weight_shift - input_phase))/np.exp(kappa)
        else:   
            for input_num, input_phase in enumerate(phases):
                self.w[:, input_num] = c_exc * np.exp( - pow((phases-input_phase),2) / (2* pow(sigma, 2))) - c_inh
                self.w_right[:, input_num] = c_exc * np.exp( - pow((phases + weight_shift - input_phase), 2) / (2 * pow(sigma, 2))) - c_inh
                self.w_left[:, input_num] = c_exc * np.exp( - pow((phases - weight_shift - input_phase), 2) / (2 * pow(sigma, 2))) - c_inh
                

    def init_network_activity(self,peak_cell_num,init_time):
        
        self.u[self.size-1-peak_cell_num] = 1       #Initial active grid cell
        self.target.set_init_mode(True)
        self.run(init_time)
        self.target.set_init_mode(False)
        self.init_time = init_time
        return 0
            
    def run(self, sim_time):
        for step_num in range(int(round(sim_time/self.dt)) + 1):
            u_out = 1/(1 + np.exp(self.beta*(self.u - 0.5)))
            self.u_out_log.append(u_out)
            
            movement_input = self.target.update_position_1d(self.dt,step_num+self.init_time/self.dt)
            
            self.u_shift   = u_out * movement_input[0]
            self.u_shift_r = u_out * movement_input[1]
            self.u_shift_l = u_out * movement_input[2]
            
            exc_input = np.dot(self.u_shift, self.w)
            exc_input_r = np.dot(self.u_shift_r,self.w_right)
            exc_input_l = np.dot(self.u_shift_l,self.w_left)
            
            if self.von_mises:
                inh_input = max(0, np.sum(u_out) - 1)               #Inhibition, needed for von Mises distribution
            else:
                inh_input = 0

            self.u += (-self.u + exc_input - inh_input - self.h + exc_input_r + exc_input_l)/self.tau*self.dt
            self.u_log.append(self.u.copy())
            
            #bin_index = int(round(self.real_position,2) *100)
            #self.spatial_bin[bin_index] += 1

        #print(self.spatial_bin)
    
    def plot_activities(self,u_out):
        fig, ax = plt.subplots()
        u = self.u_log
        if u_out:
            u = self.u_out_log
        u = np.array(u)
        mat = ax.matshow(u.transpose(), aspect="auto",
                         extent=(-self.dt/2, u.shape[0]*self.dt - self.dt/2, -0.5, u.shape[1]-0.5))
        ax.xaxis.set_ticks_position('bottom')
        ax.set_title("Network activities")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Unit number")
        target_data = self.target.fetch_log_data()
        x_val = [x[0] for x in target_data]
        y_val = [x[1] for x in target_data]
        plt.plot(x_val,y_val,'r-')
        cbaxes = fig.add_axes([0.92, 0.1, 0.01, 0.8])
        plt.colorbar(mat, cax=cbaxes)
        ax2 = ax.twinx()
        ax2.set_ylim(0,self.size * self.cell_distance)
        ax2.set_ylabel("Distance (m)")
        plt.subplots_adjust(right=0.8)

    def slope_accuracy(self,speed,sim_time,peak_cell_num):
        travel_distance = sim_time * speed
        single_cell_activity = []
        for value in self.u_log:
            single_cell_activity.append(value[self.size - 1])
        sca = single_cell_activity[int(self.init_time/self.dt):len(single_cell_activity)]
        step_size = travel_distance/(len(self.u_log) - 1 - self.init_time/self.dt)
        peaks, _ = find_peaks(sca, height=0.5)
        if len(peaks) == 0:
            return math.nan
        a = np.array([[0,peaks[0]*self.dt],[peak_cell_num,0]])
        cell_reg = stats.linregress(a)
        #print('Activity Speed '+str(cell_reg.slope*self.cell_distance))
        #print('Real Speed'+str(-speed))
        #print('Slope accuracy (%):')
        result = (cell_reg.slope - (-speed)/self.cell_distance)/(-speed/self.cell_distance)*100
        return result
        

    def plot_single_cell(self,speed,sim_time,index):
        travel_distance = sim_time * speed
        single_cell_activity = []
        for value in self.u_log:
            single_cell_activity.append(value[self.size -1 -index])
        sca = single_cell_activity[int(self.init_time/self.dt):len(single_cell_activity)]
        step_size = travel_distance/(len(self.u_log) - 1 - self.init_time/self.dt)
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position('bottom')
        ax.set_title("Cell #"+str(self.size -1 -index)+" Activity")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Activity Value")
        x_val = np.arange(0,travel_distance+step_size,step_size)
        if len(x_val) > len(sca):
            x_val = np.arange(0,travel_distance+step_size/2,step_size)
        plt.plot(x_val,sca,'b-')
        peaks, _ = find_peaks(sca, height=0.5)
        if len(peaks) > 0:
            plt.plot(peaks[0]*self.dt*speed,sca[peaks[0]],"x")
        
        #print("Average cell activity:")
        #print(np.sum(single_cell_activity)*self.dt/sim_time)
        
##        for i in range(len(self.u_log)):
##            s = int(round((x_val[i]*100),0) % (self.size*self.grid_position_factor *100 -1)) 
##            if s == 200:
##                print(x_val[i])
##            self.spatial_bin[s] += 1
##            self.activity_sums[s] += single_cell_activity[i]
##        print('Spatial bins [1cm]')
##        print(self.spatial_bin)
##        print('Summed activities')
##        print(self.activity_sums)


    
