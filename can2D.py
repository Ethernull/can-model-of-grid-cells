import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.widgets import Slider
from scipy.signal import find_peaks
from target2D import TARGET2D

class CAN2D:
    def __init__(self,target,cells_per_row,cell_rows,init_activity,dt,beta,ws_param,tau,h,period_length):
        self.target = target                #Target object, providing movement data [speed, direction]
        self.x = cells_per_row              #Number of cells per row
        self.y = cell_rows                  #Number of cell rows
        self.size = self.x * self.y         #Number of total cells
        self.period_length = period_length  #Real life distance projected on grid horizontally

        self.u = np.zeros(self.size)        #Cell activities
        self.cell_x = np.zeros(self.size)   #Horizontal positions of cells
        self.cell_y = np.zeros(self.size)   #Vertical positions of cells
        

        self.u_out_log = []             #Log for grid cell activity values after calculations
        self.u_log = []                 #Log for grid cell activity values before calculations, after applying sigmoid

        self.dt = dt                #Delta time
        self.beta = beta            #Sigmoid factor
        self.ws_param = ws_param    #Weight shift parameter
        self.tau = tau              #Time constant
        self.h = h                  #Adjustment factor

        self.i_pos = init_activity  #Index number of first active cell
        self.u[self.i_pos] = 1

        self.init_time = 0          #Time spent in initialization phase

        #Creating positions for cells
        for j in range(0,self.y):
            for i in range(0,self.x):
                index = i+(self.x*j)
                self.cell_x[index] = (i+0.5)/self.x
                self.cell_y[index] = (j+0.5)/self.y

        #Triangular height correction
        self.cell_y = self.cell_y * (math.sqrt(3)/2)

        #Places target at initial network spike to sync target and network
        self.target.set_start_pos([self.cell_x[self.i_pos]*self.period_length,self.cell_y[self.i_pos]*self.period_length])

        #Cell-to-Cell distance calculations---------------------------
        self.distance = self.calc_distances((0,0))
        self.distance_r = self.calc_distances(self.target.angle_to_vector(0))
        self.distance_t_r = self.calc_distances(self.target.angle_to_vector(60))
        self.distance_t_l = self.calc_distances(self.target.angle_to_vector(120))
        self.distance_l = self.calc_distances(self.target.angle_to_vector(180))
        self.distance_b_l = self.calc_distances(self.target.angle_to_vector(240))
        self.distance_b_r = self.calc_distances(self.target.angle_to_vector(300))

        #Weight calculations-------------------------------------------
        self.weights = self.calc_weights(self.distance)
        self.weights_r = self.calc_weights(self.distance_r)
        self.weights_t_r = self.calc_weights(self.distance_t_r)
        self.weights_t_l = self.calc_weights(self.distance_t_l)
        self.weights_l = self.calc_weights(self.distance_l)
        self.weights_b_l = self.calc_weights(self.distance_b_l)
        self.weights_b_r = self.calc_weights(self.distance_b_r)

    #Euclidean distance function including twisted torus attributes and a shifting mechanism
    def min_distance(self,a,b,shift_dir):
        d = []
        #Triangular height
        tri_h = math.sqrt(3)/2
        #Twisted torus attributes
        s = [[0,0],[-0.5,tri_h],[-0.5,-tri_h],[0.5,tri_h],[0.5,-tri_h],[-1,0],[1,0]]
        
        shift_strength = self.ws_param* (-1/self.x)
        shift_x = shift_dir[0] * shift_strength
        shift_y = shift_dir[1] * shift_strength

        for si in s:
            di = math.sqrt((b[0]-a[0]+si[0]+shift_x)**2 +(b[1]-a[1]+si[1]+shift_y)**2)
            d.append(di)
        return np.amin(d)
    
    #Calls distance function for each possible cell pair and returns collection of results            
    def calc_distances(self,shift):
            dist = np.zeros((self.size,self.size))
            for a in range(0,self.size):
                dist[a,a] = 0
                for b in range(0,self.size):
                    cell_a = ([self.cell_x[a],self.cell_y[a]])
                    cell_b = ([self.cell_x[b],self.cell_y[b]])
                    result = self.min_distance(cell_a,cell_b,shift)
                    dist[a,b] = result
            return dist

    #Calculates weights using a gaussian function
    def calc_weights(self,distances):
        weights = np.empty((self.size,self.size))
        I = 1
        sigma = 1
        for a in range(0,self.size):
            for b in range(0,self.size):
                result = I*np.exp(- (distances[a,b]**2)/(sigma**2))
                weights[a,b] = result
        return weights

    #Initilizes network by running it non-moving target
    def init_network(self,init_time):
        #Init mode: Target returns speed: 0, direction: standing
        self.target.set_init_mode(True)
        self.run(init_time)
        self.init_time += init_time
        self.target.set_init_mode(False)

    def run(self, sim_time):
        for step_num in range(int(round(sim_time/self.dt)) + 1):
            #Applying Sigmoid
            u_out = 1/(1 + np.exp(self.beta*(self.u - 0.5)))
            self.u_out_log.append(u_out)

            #Updating and aquiring target movement and direction data
            movement_input = self.target.update_position_2d(self.dt,step_num)

            #Calculating shifting layers
            self.u_shift   = u_out * movement_input[0]
            self.u_shift_r = u_out * movement_input[1]
            self.u_shift_t_r = u_out * movement_input[2]
            self.u_shift_t_l = u_out * movement_input[3]
            self.u_shift_l = u_out * movement_input[4]
            self.u_shift_b_l = u_out * movement_input[5]
            self.u_shift_b_r = u_out * movement_input[6]

            ##Calculating excitation values with shifting layers and corresponding weight matrices
            exc_input = np.dot(self.u_shift, self.weights)
            exc_input_r = np.dot(self.u_shift_r,self.weights_r)
            exc_input_t_r = np.dot(self.u_shift_t_r,self.weights_t_r)
            exc_input_t_l = np.dot(self.u_shift_t_l,self.weights_t_l)
            exc_input_l = np.dot(self.u_shift_l,self.weights_l)
            exc_input_b_l = np.dot(self.u_shift_b_l,self.weights_b_l)
            exc_input_b_r = np.dot(self.u_shift_b_r,self.weights_b_r)

            exc_sum = exc_input + exc_input_r + exc_input_t_r + exc_input_t_l
            exc_sum += exc_input_l + exc_input_b_l + exc_input_b_r
            inh_input = max(0, np.sum(u_out) - 1)

            #Calculating new activity values
            self.u += (-self.u + exc_sum - inh_input - self.h)/self.tau*self.dt
            self.u_log.append(self.u.copy())
            
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
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Unit number")
        plt.show()

    def plot_activities_interactive(self):
        target_data = self.target.fetch_log_data()
        
        dx, dy = self.period_length/self.x, self.period_length*(math.sqrt(3)/2)/self.y
        y, x = np.mgrid[slice(0, self.period_length*(math.sqrt(3)/2) + dy, dy),
                        slice(0, self.period_length + dx, dx)]
        
        z = self.u_out_log[0]
        z = np.reshape(z,(self.y,self.x))
        z_min, z_max = -1, 1
        
        fig,ax = plt.subplots()
        plt.subplots_adjust(left=0.1,bottom=0.22)
        plt.axis([x.min(), x.max(), y.min(), y.max()])
        plt.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        cbar = plt.colorbar()
        cbar.set_label("Cell Activity")

        def setup():
            ax.clear()
            ax.set_title("Cell Activity at "+str(self.target.speed)+" m/s" )
            ax.set_xlabel("Distance [m]")
            ax.set_ylabel("Distance [m]")
            
        axSlider = plt.axes([0.12,0.05,0.8,0.05])
        slider = Slider(axSlider,'Time [s]',0.0,self.dt*len(self.u_out_log)-self.dt,valinit=0.00,valfmt='%1.2f',valstep=self.dt)

        #Called on slider change
        def update(val):
            index = int(val/self.dt)
            z_new = self.u_out_log[index]
            z_new = np.reshape(z_new,(self.y,self.x))
            setup()
            ax.pcolor(x, y, z_new, cmap='RdBu', vmin=z_min, vmax=z_max)
            target_pos = target_data[index]
            ax.plot(target_pos[0],target_pos[1],'ro')
            fig.canvas.draw_idle()
            
        update(0)
        slider.on_changed(update)
        plt.show()

    def plot_single_cell(self,speed,sim_time,index):
        init_time = self.init_time
        travel_distance = sim_time * speed
        single_cell_activity = []
        for value in self.u_log:
            single_cell_activity.append(value[index])
        sca = single_cell_activity[int(init_time/self.dt):len(single_cell_activity)]
        step_size = travel_distance/(len(self.u_log) - 1 - init_time/self.dt)
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position('bottom')
        ax.set_title("Cell #"+str(index)+" Activity")
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Activity Value")
        x_val = np.arange(0,travel_distance+step_size,step_size)
        if len(x_val) > len(sca):
            x_val = np.arange(0,travel_distance+step_size/2,step_size)
        plt.plot(x_val,sca,'b-')
        peaks, _ = find_peaks(sca, height=0.5)
        if len(peaks) > 0:
            plt.plot(peaks[0]*self.dt*speed,sca[peaks[0]],"x")
        #plt.show()

    #Network error function. Compares speed and direction of network and target
    #Implemented for  specific test case:
    #-Target starts at cell 0 [Bottom-Left]
    #-Target only moves up and/or to the right [0°-90°]
    def error_benchmark(self,print_details):
        #Index offset to exclude network initilization phase
        index_excl_init = int(self.init_time/self.dt)
        
        network_data = self.u_out_log[index_excl_init:]
        target_data = self.target.fetch_real_data()[index_excl_init:]

        #Cell index for bottom left cell
        start = 0
        #Indizes for cells at the right end of the benchmark path
        ends_right = range(self.x-1,self.size,self.x)
        #Indizes for cells at the top end of the benchmark path
        ends_top = range(self.size - self.x,self.size,1)

        #Returns time step and cell index when peak reaches specified benchmark borders
        def find_time_step():
            for index,n in enumerate(network_data):
                peak = np.argmax(n)
                if peak in ends_right:
                    return [index,peak]
                if peak in ends_top:
                    return [index,peak]
            return None
        
        result = find_time_step()
        
        if result is not None:
            time_step = result[0]
            cell_index = result[1]
            
            network_vec = [self.cell_x[cell_index] - self.cell_x[start], self.cell_y[cell_index] - self.cell_y[start]]
            target_vec = target_data[time_step]

            distance_n = self.target.vector_magnitude(network_vec) * self.period_length
            distance_t = self.target.vector_magnitude(target_vec)
            
            error = ((distance_n - distance_t)/distance_t)*100
            angle = self.target.angle_between_vectors(network_vec,target_vec)
            if print_details:
                print("Network Distance:")
                print(distance_n)
                print("Target Distance:")
                print(distance_t)
                print("Error % [ND-TD/TD]:")
                print(error)
                print("Angle between direction vectors:")
                print(angle)
            return error
        else:
            print("Accurracy check failed")
            return math.nan
        
