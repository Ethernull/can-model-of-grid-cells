import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import vonmises
from scipy.special import i0

cellValues = [1.0,0.5,0.333,0.25,0.25,0.333,0.5]

cellArray = np.array(cellValues)

startingPosition = 0

#TODO implement weight matrix

#TODO implement a von Mises distribution and apply to weight matrix

mu, kappa = 0.0, 4.0 # mean and dispersion

s = np.random.vonmises(mu, kappa, 1000) #samples

xVM = np.linspace(-np.pi, np.pi, num=7)

yVM = np.exp(kappa*np.cos(xVM-mu))/(2*np.pi*i0(kappa))

#plt.plot(xVM, yVM, linewidth=2, color='r')

#plt.show()


def updateCellValues(position):
    #dt = 0.0005s  //
    #theta = 0.01s //from Burak, 2009
    for i in range(len(cellArray)):
        #si = array[i]
        #dsi = (placeholderF(position,i) - si)*dt / theta
        #si += dsi
        #array[i] = si
        cellArray[i] = placeholderFunction(position,i)
    print(cellArray,type(cellArray))


#TODO implement differential equotation for activation of units
def placeholderFunction(x,y):
    return 1/(cellProximity(x,y)+1)

#TODO replace with weight system
def cellProximity(positionX,positionY):
    threshold = len(cellValues)/2
    difference = abs(positionX-positionY)
    if difference > threshold:
        return len(cellValues)-difference
    else:
        return difference

        
#TODO implement better simulation for animal movement
def randomMovement(position):
    x = random.randrange(2)
    if x == 0:
        position += 1
        
    if x == 1:
        position -= 1

    #ring shape adjustments
    if position < 0:
        position = len(cellArray)-1
        
    if position == len(cellArray):
        position = 0

    return position


def plotActivity(position):
    plt.axis([0,len(cellArray)-1,0,1])

    for t in range(20):
        plt.clf()
        plt.plot(cellArray)
        position = randomMovement(position)
        updateCellValues(position)
        plt.pause(2)
        
    plt.show()


plotActivity(startingPosition)

