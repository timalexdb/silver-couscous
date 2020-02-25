# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:54:14 2020

@author: Timot
"""
import networkx as nx  
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
import pandas as pd
import random
import pickle
from matplotlib import cm
from UltimatumGame import Agent, Graph
#from collections import OrderedDict
#from colorspacious import cspace_converter


graphType = ['Watts-Strogatz', 'Barabasi-Albert']
agentList = []

set_self = True

if set_self:
    simulations = 40
    rounds = 100
    agentCount = 12
    edgeDegree = 5
    
    explore = 0.4
    
    proportional = True
    randomPlay = True
    
    showGraph = False
    focus = None

#else:
#    simulations, rounds, agentCount, edgeDegree, explore, proportional, randomPlay = settings
    
g = "Barabasi-Albert"
sim = 30

with open("Data/{0}V{1}_n{2}_sim{3}_round{4}_exp={5:.2f}_prop={6}_random={7}.pickle"
          .format(g, sim, agentCount, simulations, rounds, explore, str(proportional), str(randomPlay)), 'rb') as f:
    Agents = pickle.load(f)

for agent in Agents:
    agentList.append(str(agent))
    print(agent.shareData())


finalDat = []


gameAna = pd.read_csv("Data/gameData_n{0}_sim{1}_round{2}_exp={3:.2f}_prop={4}_random={5}.csv".format(agentCount, simulations, rounds, explore, str(proportional), str(randomPlay)), encoding="utf-8", header = [0,1,2])
edgeAna = pd.read_csv("Data/edgeData_n{0}_sim{1}_round{2}_exp={3:.2f}_prop={4}_random={5}.csv".format(agentCount, simulations, rounds, explore, str(proportional), str(randomPlay)), encoding='utf-8', header = [0,1,2])


print(edgeAna)
"""
for g in graphType:

    for sim in range(simulations):

        read = open("Graphs/{0}V{1}_n{2}_sim{3}_round{4}_exp={5:.2f}_prop={6}_random={7}.gpickle".format(g, sim, agentCount, simulations, rounds, explore, str(proportional), str(randomPlay)), 'rb')
        graph = nx.read_gpickle(read)#, nodetype = int)

        if showGraph:
            nx.draw(graph, node_color='r', with_labels=True, alpha=0.53, width=1.5)
            plt.title('{0} (simulation {1}/{2})'.format(g, sim+1, simulations))
            plt.show()
"""

readFocus = open("Graphs/{0}V10_n{2}_sim{3}_round{4}_exp={5:.2f}_prop={6}_random={7}.gpickle"
                 .format(g, sim, agentCount, simulations, rounds, explore, str(proportional), str(randomPlay)), 'rb') #{1}_n{2}_sim{3}_round{4}_exp={5:.2f}_prop={6}_random={7}".format(g, sim, agentCount, simulations, rounds, explore, str(proportional), str(randomPlay)))
graphFocus = nx.read_gpickle(readFocus)#, nodetype=int)



graph = graphFocus
positions = nx.spring_layout(graph)

# Animation function

#def color_calc():
#    for n in range(rounds):

def size_calc(i):
    paylist = []
    size = 1200
    dev = 0.5
    size_map = [size] * len(agentList)
    
    for agent in Agents:
        #print("Payoff for {0}: {1}".format(agent, agent.shareData()[i][2]))
        paylist.append((agent.shareData()[i][2]))
    
    roundMax = max(paylist)#, key = lambda payoff : payoff[1])
    
    mean = np.mean(paylist)
    stdev = np.std(paylist)
    
    for val in paylist:
        newsize = round(size + ((size*dev) * ((val-mean) / stdev)), 0)
        size_map[paylist.index(val)] = newsize
    
    #print(paylist)
    #print("roundMax: {0}".format(roundMax))
    #print("before: %s" % size_map)
    #print("after: %s" % size_map)
        
    return(size_map)

#size_calc(5)

def nodeCol_calc(i):
    walletlist = []
    color_map = walletlist
    
    stratlist = []
    for agent in Agents:
        walletlist.append(np.sum(agent.wallet[:i+1]))
        stratlist.append(agent.shareData()[i])
    print(stratlist)
    
    mean = np.mean(walletlist)
    stdev = np.std(walletlist)
    
    for val in color_map:
        z = round((val-mean)/stdev, 2)
        color_map[color_map.index(val)] = z #round(mean + ((mean*stdev) * ((val-mean) / stdev)), 0)
    #print(walletlist)
    return(color_map)

size_calc(5)
nodeCol_calc(5)

for i in range(20):
    nodeCol_calc(i)

def animate(i):
    plt.clf()
    nx.draw(graph, pos = positions, node_color = nodeCol_calc(i), alpha = 0.53, node_size = size_calc(i), width = 1.5, cmap = plt.cm.seismic, with_labels=True, font_size = 30)
    # node_color=[random.choice(color_map) for j in range(len(agentList))]
    # add _r to color to get reverse colormap

#nx.draw(graph)
fig = plt.figure(figsize=(10,8))
anim = ani.FuncAnimation(fig, animate, frames = rounds, interval = 200, repeat_delay = 2000)#, blit=True)


#for i in range(20):
#    nodeCol_calc(i)


"""
for edge in graphFocus.edges:
    print(list(graphFocus.get_edge_data(*edge).values()))
    
print("\n")

for node in graphFocus.nodes:
    print(graphFocus.nodes[node]['agent'])
    print(graphFocus.nodes[node]['agent'].shareData())
    #print(nx.get_node_attributes(graphFocus, 'agent'))
"""

        
# either use gameAna['graphtype']['sim']['agent(s)'][row:row+1] or gameAna[loc], see https://stackoverflow.com/questions/53927460/select-rows-in-pandas-multiindex-dataframe

        
        

"""
def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

fig1 = plt.figure()

# Fixing random state for reproducibility
np.random.seed(19680801)

data = np.random.rand(2, 25)
l, = plt.plot([], [], 'r-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x')
plt.title('test')
line_ani = ani.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
                                   interval=50, blit=True)

fig2 = plt.figure()

x = np.arange(-9, 10)
y = np.arange(-9, 10).reshape(-1, 1)
base = np.hypot(x, y)
ims = []
for add in np.arange(15):
    ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))

im_ani = ani.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
                                   blit=True)

plt.show()
fig2
"""

#   to change IPython backend, enter %matplotlib followed by 'qt' for animations or 'inline' for plot pane
