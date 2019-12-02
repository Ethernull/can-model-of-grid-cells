import numpy as np
import matplotlib.pyplot as plt


class CAN:
    def __init__(self, size, tau, dt, kappa, beta,movement_mode, plot_weights=True):
        #TODO implement activation function from Sanarmirskaya and Schöner - 2010
        self.u = np.random.random(size) #activation function (stabilization factor if *(-1)) ?
        self.u_log = []
        self.tau = tau
        self.dt = dt
        self.beta = beta
        self.u_shift_none = self.u
        self.u_shift_right = self.u
        self.u_shift_left = self.u
        self.mode = movement_mode #0: none #1: right

        # create weight matrix
        self.w = np.empty((size, size))
        phases = np.arange(-np.pi, np.pi, 2*np.pi/size)
        #for input_num, input_phase in enumerate(phases):
            #self.w[:, input_num] = np.exp(kappa * np.cos(phases - input_phase))/np.exp(kappa)

        #TODO integrate inhibitory connections
        #range : sigma (3.0 ; [3.0,3.0])
        #excitatory connectivity strength: c_exc (0.9 ; 0.6)
        #inhibitory connectivity strength: c_inh (5.0 ; 0.5)
        #values used in Sanarmirskaya and Schöner - 2010 

        sigma = 3.0
        c_exc = 0.9
        c_inh = 5.0

        #TODO create shifted weight distributions based on needed shifting directions
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

    def run(self, sim_time):
        for step_num in range(int(round(sim_time/self.dt)) + 1):
            u_out = 1/(1 + np.exp(self.beta*(self.u - 0.5)))   #Sigmoid function (Beta = -8)(? = 0.5)
            exc_input = np.dot(u_out, self.w)           #Sum: f(u(x, t)ω(x − x')dx'
            inh_input = max(0, np.sum(u_out) - 1)       #External input S(x,t) ?

            
            #TODO implement negative resting level h < 0
            #TODO use shifting layers based on return value of movement_signal and gridcell input
            self.u += (-self.u + exc_input - inh_input)/self.tau*self.dt
            self.u_log.append(self.u.copy())

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

    def no_movement():
        return 0

    def moving_right():
        return 1
    
    def movement_signal():
        switcher = {
            1:no_movement,
            2:moving_right
            #3:moving left,
            #4:random_movement,
        }

        func = switcher.get(mode, lambda: "Invalid mode")

        func()
