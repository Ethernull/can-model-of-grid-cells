from can import CAN
from target import TARGET
import matplotlib.pyplot as plt

peak_cell_num = 15

tg = TARGET(speed=0.6,max_speed=0.8,size=20,peak_cell_num=peak_cell_num)

can = CAN(target=tg,size=20,tau=0.041, dt=0.01,kappa=0.1,beta =-8,ws_param=0.06, h=0)
can.init_network_activity(peak_cell_num=peak_cell_num,init_time=1)
can.run(sim_time=20)
can.plot_activities(u_out=True)

print(can.slope_accuracy(0.1))

plt.show()
