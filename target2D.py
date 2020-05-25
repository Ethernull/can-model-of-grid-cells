import math
import numpy as np

class TARGET2D:
    def __init__(self, speed, max_speed,direction,period_length):
        self.speed = speed
        self.max_speed = max_speed
        self.period_length = period_length

        self.grid_position = []
        self.grid_position_log = []
        self.real_position = []
        self.real_position_log = []
        self.distance = 0
        self.distance_log = []
        
        self.init_mode = False;
        self.tri_h = math.sqrt(3)/2

        #Directional vector calculated from angle in degrees
        #0° equals a direction vector of (1,0)
        #90° equals (0,1) etc
        self.v = self.angle_to_vector(direction)
        
        #Hexa-directional vector for grid calculations
        #Contains a value between 0 and 1 for each shifting layer of the network:
        #standing,right,top right, top left, left, bottom left, bottom right
        self.hexa_v = self.angle_to_hexa_v(direction)

    def signal_amplifier(self):
        #amp_sig = 0.5* (-abs(self.speed) + 1.6)
        amp_sig = (-0.1/0.6) * self.speed + 1.12
        #return amp_sig
        return 1
        
    def angle_to_vector(self,angle):
        ang = math.radians(angle)
        v = np.array([math.cos(ang),math.sin(ang)])
        return v
    
    def angle_to_hexa_v(self,angle):
        movement_strength = self.speed/self.max_speed
        standing = 1 - movement_strength
        right = 0
        top_r = 0
        top_l = 0
        left  = 0
        bot_l = 0   
        bot_r = 0
        
        if angle == 0:
            right = 1
        elif angle < 60:
            right = (60-angle)/60
            top_r = 1 - right
            #right = math.cos(math.radians(angle - 0))
            #top_r = math.cos(math.radians(60 - angle))
            #print(right)
            #print(top_r)
        elif angle == 60:
            top_r = 1
        elif angle < 120:
            top_r = (120-angle)/60
            top_l = 1 - top_r
        elif angle == 120:
            top_l = 1
        elif angle < 180:
            top_l = (180-angle)/60
            left = 1 - top_l
        elif angle == 180:
            left = 1
        elif angle < 240:
            left = (240-angle)/60
            bot_l = 1 - left
        elif angle == 240:
            bot_l = 1
        elif angle < 300:
            bot_l = (300-angle)/60
            bot_r = 1 - bot_l
        elif angle == 300:
            bot_r = 1
        elif angle < 360:
            bot_r = (360-angle)/60
            r = 1 - bot_r
        else:
            r = 1

        hexa_v =  np.array([0,right,top_r,top_l,left,bot_l,bot_r])
        hexa_v = hexa_v * movement_strength * self.signal_amplifier()
        hexa_v[0] = standing
        return hexa_v

    def set_start_pos(self,s):
        self.grid_position = s
        self.grid_position_log.append([self.grid_position[0],self.grid_position[1]])
        self.real_position = [0,0]
        self.real_position_log.append([self.real_position[0],self.real_position[1]])
    
    def update_position_2d(self,dt,step_num):
        if self.init_mode:
            self.grid_position_log.append([self.grid_position[0],self.grid_position[1]])
            self.real_position_log.append([self.real_position[0],self.real_position[1]])
            return [1,0,0,0,0,0,0]
        
        speed_vector = self.speed * dt * self.v

        self.grid_position += speed_vector

        self.real_position += speed_vector

        self.distance += math.sqrt((speed_vector[0])**2 +(speed_vector[1])**2)
        
        #Out of bounds calculations [Twisted Torus]
        #Right side
        if self.grid_position[0] > self.period_length:
            self.grid_position = [self.grid_position[0]-self.period_length,self.grid_position[1]]
        #Left side
        if self.grid_position[0] < 0:
            self.grid_position = [self.grid_position[0]+self.period_length,self.grid_position[1]]                      
        #Top side, left half
        if self.grid_position[1] > self.tri_h*self.period_length:
            x = self.grid_position[0]+0.5*self.period_length
            #Top side, right half
            if x > 1*self.period_length:
                x -= 1*self.period_length
            self.grid_position = [x,self.grid_position[1]-self.tri_h*self.period_length]
        #Bottom side, left half
        if self.grid_position[1] < 0:
            x = self.grid_position[0]+0.5
            #Bottom side, right half
            if x > 1*self.period_length:
                x -= 1*self.period_length
            self.grid_position = [x,self.grid_position[1]+self.tri_h]

        self.grid_position_log.append([self.grid_position[0],self.grid_position[1]])
        self.real_position_log.append([self.real_position[0],self.real_position[1]])
        self.distance_log.append(self.distance)

        return self.hexa_v

    def fetch_log_data(self):
        return self.grid_position_log

    def fetch_real_data(self):
        return self.real_position_log

    def vector_magnitude(self,vector):
        return math.sqrt((vector[0])**2 +(vector[1])**2)

    def angle_between_vectors(self,vec1,vec2):
        dot_p = np.dot(vec1,vec2)
        mag1 = self.vector_magnitude(vec1)
        mag2 = self.vector_magnitude(vec2)
        return math.degrees(math.acos(dot_p/(mag1*mag2)))

    def set_init_mode(self,init_active):
        self.init_mode = init_active
