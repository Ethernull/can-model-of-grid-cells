from can import CAN


can = CAN(size=20, tau=8, dt=0.1, kappa=4, beta=-8, h=0,movement_mode=0,von_mises=False)
can.run(sim_time=100)
can.plot_activities()