from can import CAN

#Initially: tau = 8, kappa = 4
can = CAN(size=20, tau=2, dt=0.1, kappa=0.5, beta=-8, h=0,movement_mode=0,von_mises=True)
can.run(sim_time=200)
can.plot_activities()
