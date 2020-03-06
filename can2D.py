import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import math
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def create_grid():
    x = np.asarray([0, 1, 2, 3, 0.5, 1.5, 2.5, 0, 1, 2, 3, 0.5, 1.5, 2.5, 0, 1, 2, 3])
    y = np.asarray([0, 0, 0, 0, 1.0, 1.0, 1.0, 2, 2, 2, 2, 3.0, 3.0, 3.0, 4, 4, 4, 4])
    y = y*math.sqrt(3)/2
    triangles = [[0, 1, 4], [4,5,1],[1,2,5],[5,6,2],[2,3,6],
                 [4,7,8],[4,8,5],[5,8,9],[5,9,6],[6,9,10],
                 [7,11,8],[8,11,12],[8,12,9],[9,12,13],[9,13,10],
                 [11,14,15],[11,15,12],[12,15,16],[12,16,13],[13,16,17]]
    triang = mtri.Triangulation(x, y, triangles)

    fig, ax = plt.subplots()
    z = np.cos(1.5 * x) * np.cos(1.5 * y)
    ax.tricontourf(triang, z)
    ax.triplot(triang, 'ko-')
    ax.set_title('Triangular grid')

def create_weight_matrix():
    a = (1,1)
    b = (1,2)
    c = (2,1)
    cell_positions = [a,b,c]
    #print(distance.euclidean(a,b))
    weights = np.empty((20,20))
    ws_param = 0
    weight_shift = 2* np.pi * ws_param
    weight_strength = 1
    phases = np.arange(-np.pi, np.pi, 2*np.pi/20)
    kappa = 0.1

    #for input_num, input_phase in enumerate(phases):
    #    weights[:, input_num] = weight_strength * np.exp(kappa * np.cos(phases - input_phase))/np.exp(kappa)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    X = phases
    Y = phases
    X, Y = np.meshgrid(X, Y)
    #R = distance.euclidean([X,Y])
    R = np.sqrt(X**2 + Y**2)
    Z = np.exp(kappa * np.cos(R))/np.exp(kappa)
    

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    
create_weight_matrix()
#create_grid()
plt.show()
