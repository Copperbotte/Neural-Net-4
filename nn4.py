
import numpy as np
import random as rand

#neural net generating functions
def _makeNNet(n1, n2, func):
    #from n1+1 to n2
    return np.array([[func(x,y) for x in range(n1+1)] for y in range(n2)])

def makeZeroNNet(n1, n2):
    def zeroFunc(x,y):
        return 0.0
    return _makeNNet(n1, n2, zeroFunc)

def makeIdentityNNet(n1, n2):
    def identityFunc(x,y):
        return 1.0 if x == y else 0.0
    return _makeNNet(n1, n2, identityFunc)

def makeRandNNet(n1, n2):
    def randFunc(x,y):
        return rand.random() * 2 - 1
    return _makeNNet(n1, n2, randFunc)

#sigmoids
def sigTanh(x):
    return np.tanh(x)

def dsigTanh(sig):
    return 1.0 - sig**2

#States is the same length as the first node in Nodes.
#Sigmoid is performed before the transform rather than after,
#   to accomodate for non-bounded neural net outputs.
def forwardprop(nnet, states):
    res = [states]
    for W in nnet:
        sig = [1.0] + list(map(sigTanh, res[-1]))#sigmoid input
        N1 = np.array([sig]).transpose()        #build vector
        N2 = W @ N1                             #mul
        res.append(N2.transpose().tolist()[0])  #un-build vector
    return res

def backprop(nnet, nodes, states, expect):
    #forwardprop
    res = [states]
    sigs = []
    for W in nnet:
        sig = [1.0] + list(map(sigTanh, res[-1]))#sigmoid input
        N1 = np.array([sig]).transpose()        #build vector
        N2 = W @ N1                             #mul
        res.append(N2.transpose().tolist()[0])  #un-build vector
        sigs.append(sig)

    #backprop
    delta = [makeZeroNNet(a,b) for a,b in zip(Nodes[:-1], Nodes[1:])]

    dN = [r-e for r,e in zip(res[-1], expect)]  #delta node
    error = sum([e**2 / 4.0 for e in dN])       #numerical error
    dNode = np.array([dN]).transpose()          #delta node vector

    for N in range(len(nnet)):
        n = len(nnet) - N - 1 #reverse order
        #find weights
        #deltaNNet is the unique combo of dN2 * sigma(N1)
        for y in range(len(delta[n])):
            for x in range(len(delta[n][0])):
                delta[n][y][x] += sigs[n][x] * dNode[y][0]

        #backprop to next node
        dNode = nnet[n].transpose() @ dNode #weight derivative
        dNode = dNode[1:]                   #strip bias
        #sigmoid derivative
        dNode *= 0.5 * np.array([list(map(dsigTanh, sigs[n][1:]))]).transpose()
    
    return res, error, delta

#Nodes includes start and end nodes
Nodes = [2, 3, 5, 3, 1]

#NNet includes a hidden bias node from its source as the 1st element.
#Each matrix transforms between two Nodes elements.
NNet = [makeRandNNet(a,b) for a,b in zip(Nodes[:-1], Nodes[1:])]

print("Nodes")
print(Nodes)
for n in NNet:
    print(n)
    print()

#make dataset
dataset = []
for d in range(100):
    x = rand.random() * 2.0 - 1.0
    y = rand.random() * 2.0 - 1.0
    c = np.sin(x*2*np.pi) * np.sin(y*2*np.pi)
    #c = -1
    #if x**2 + y**2 < 0.5:
    #    c = 1
    dataset.append([x,y,c])

import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())
import optical_tools as ot
import time

img = np.array([[ot.l2s([x/100,y/100,0]) for x in range(100)] for y in range(100)])
#plt.imshow(img)
#plt.show()

prevBigError = 1000
prevtime = time.time()
lastError = 1000000

def BPnDisp(arg=""):

    E = 10000
    #while E <
    
    global NNet
    global Nodes
    global prevBigError
    global prevtime
    global lastError
    rate = 0.5
    #TempNNet = [N.copy() for N in NNet]
    #while True:
    for n in range(10):
        #TempNNet = [N.copy() for N in NNet]
        Bigdelta = [makeZeroNNet(a,b) for a,b in zip(Nodes[:-1], Nodes[1:])]
        Bigerr = 0
        for d in dataset:
            Result, Error, Delta = backprop(NNet, Nodes, d[:2], d[2:])
            for n in range(len(NNet)):
                Bigdelta[n] += Delta[n] / len(dataset)
            Bigerr += Error / len(dataset)
        for n in range(len(NNet)):
            NNet[n] -= rate * Bigdelta[n]
    lastError = Bigerr
    #NNet = TempNNet
    
    print(Bigerr)
    prevtime = time.time()
    prevBigError = Bigerr
    img = []
    for y in range(100):
        line = []
        Y = -2.0*(y/100) + 1.0
        for x in range(100):
            X = 2.0*(x/100) - 1.0
            Result = forwardprop(NNet, [X, Y])[-1][0]
            c = np.tanh(Result)
            color = [c,0,0]
            if c < 0:
                color = [0,-c,0]
            line.append(ot.l2s(color))
        img.append(line)
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.clf()
    plt.scatter([d[0]*50 + 50 for d in dataset], [d[1]*50 + 50 for d in dataset])
    plt.imshow(img)
    if arg == "show":
        plt.show()
    else:
        plt.draw()
    plt.pause(0.00001)



plt.ion()
#fig = plt.figure()
#win = fig.canvas.manager.window
#win.after(100, animate, 1)
#BPnDisp()


#img = []
#for y in range(100):
#    line = []
#    Y = -2.0*(y/100) + 1.0
#    for x in range(100):
#        X = 2.0*(x/100) - 1.0
#        Result = np.sin(X*2*np.pi) * np.sin(Y*2*np.pi)
#        c = np.tanh(Result)
#        color = [c,0,0]
#        if c < 0:
#            color = [0,-c,0]
#        line.append(ot.l2s(color))
#    img.append(line)
#plt.imshow(img)
#plt.plot()
#plt.pause(10)

stop = False
def on_close(evt):
    global stop
    print("Nodes")
    print(Nodes)
    for n in NNet:
        print(n)
        print()
    stop = True

plt.figure().canvas.mpl_connect('close_event', on_close)

while not stop:
    BPnDisp()





