import math

class TARGET:
    def __init__(self, speed, max_speed, size, peak_cell_num, period_length=0.5):
        self.speed = speed
        self.max_speed = max_speed
        self.size = size
        self.cell_distance = period_length/size

        self.real_position = peak_cell_num * self.cell_distance
        self.grid_position_log = []
        self.init_mode = False;

    def update_position_1d(self,dt,step_num):
        if self.init_mode:
            self.grid_position_log.append((step_num*dt,self.real_position/self.cell_distance))
            return [1,0,0]

        self.real_position -= self.speed * dt
        self.grid_position_log.append((step_num*dt,self.real_position/self.cell_distance))

        if self.real_position < 0:
            self.grid_position_log.append((step_num*dt,math.nan))
            self.real_position = (self.size - 1) * self.cell_distance

        if self.real_position > (self.size - 1) * self.cell_distance:
            self.grid_position_log.append((step_num*dt,math.nan))
            self.real_position = 0

        s = 0
        r = 0
        l = 0
        
        if self.speed > 0:
            r = self.speed/self.max_speed
            s = 1 - r
        elif self.speed < 0:
            l = self.speed/self.max_speed
            s = 1 - l
        else:
            s = 1

        return [s,r,l]

    def fetch_log_data(self):
        return self.grid_position_log

    def set_init_mode(self,init_active):
        self.init_mode = init_active
        
