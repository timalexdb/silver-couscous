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
    simulations = 40#4
    rounds = 100#20
    agentCount = 12
    edgeDegree = 5
    
    selectionStyle = "Fermi"      # 0: Unconditional, 1: Proportional, 2: Fermi-equation
    selectionIntensity = 10 # the bèta in the Fermi-equation
    
    explore = 0.2       # with prob [explore], agents adapt strategy randomly. prob [1 - explore] = unconditional/proportional imit
    
    randomPlay = True
    
    testCase = False

if testCase:
    graphType = ['Testcase']

#else:
#    simulations, rounds, agentCount, edgeDegree, explore, proportional, randomPlay = settings
    
g = "Barabasi-Albert"
sim = 12

if testCase:
    g = 'testCase'
    sim = 3
    agentCount = 2
    edgeDegree = 1

with open("Data/{0}V{1}_n{2}_sim{3}_round{4}_exp={5:.2f}_random={6}_select={7}_beta={8}.pickle".format(g, sim, agentCount, simulations, rounds, explore, str(randomPlay), selectionStyle, selectionIntensity), 'rb') as f:
    Agents = pickle.load(f)

for agent in Agents:
    agentList.append(str(agent))
    print(agent.shareData())


finalDat = []

if testCase:
    gameAna = pd.read_csv("Data/gameTest_n{0}_sim{1}_round{2}_exp={3:.2f}_random={4}_select={5}_beta={6}.csv".format(agentCount, simulations, rounds, explore, str(randomPlay), selectionStyle, selectionIntensity), encoding='utf-8', header = [0,1,2])
    edgeAna = pd.read_csv("Data/edgeTest_n{0}_sim{1}_round{2}_exp={3:.2f}_random={4}_select={5}_beta={6}.csv".format(agentCount, simulations, rounds, explore, str(randomPlay), selectionStyle, selectionIntensity), encoding='utf-8', header = [0,1,2])
else:
    gameAna = pd.read_csv("Data/gameData_n{0}_sim{1}_round{2}_exp={3:.2f}_random={4}_select={5}_beta={6}.csv".format(agentCount, simulations, rounds, explore, str(randomPlay), selectionStyle, selectionIntensity), encoding="utf-8", header = [0,1,2])
    edgeAna = pd.read_csv("Data/edgeData_n{0}_sim{1}_round{2}_exp={3:.2f}_random={4}_select={5}_beta={6}.csv".format(agentCount, simulations, rounds, explore, str(randomPlay), selectionStyle, selectionIntensity), encoding='utf-8', header = [0,1,2])

print(gameAna)
#print(edgeAna)

readFocus = open("Graphs/{0}V{1}_n{2}_sim{3}_round{4}_exp={5:.2f}_random={6}.gpickle"
                 .format(g, sim, agentCount, simulations, rounds, explore, str(randomPlay)), 'rb') #{1}_n{2}_sim{3}_round{4}_exp={5:.2f}_prop={6}_random={7}".format(g, sim, agentCount, simulations, rounds, explore, str(randomPlay)))
graphFocus = nx.read_gpickle(readFocus)#, nodetype=int)



graph = graphFocus
positions = nx.spring_layout(graph)

# =============================================================================
# # Animation function
# =============================================================================



def safe_div(x, y):
    return 0 if y == 0 else x / y



def stratcalc():
    avgOffer = []#, avgAccept, avgSucc, strategies = ([], ) * 4
    avgAccept = []
    avgSucc = []
    
    edgeCount = len(graph.edges)
    
    for rnd in range(rounds):
        totSucc = []
        stratlist = pd.eval(gameAna[g][str(sim)].loc[rnd, :].values)
        offers, accepts = zip(*[strat[:-1] for strat in stratlist])
    
        for i in range(2):
            totSucc.extend([dat[i][2] for dat in pd.eval(edgeAna[g][str(sim)].loc[rnd, :].values)])
            # pd.eval(edgeAna[g][str(sim)].loc[0, :].values)[0,1,2] (from 1st edge 2nd list 3rd value)
        strategies.append
        
        y1 = np.sum(offers)/agentCount # must be set of
        y2 = np.sum(accepts)/agentCount
        y3 = np.sum(totSucc)/(edgeCount*2) #every edge is played twice
        
        avgOffer.append(y1)
        avgAccept.append(y2)
        avgSucc.append(y3)


    return(avgOffer, avgAccept, avgSucc, stratlist)
    
offerlist, acceptlist, successlist, stratlist = stratcalc()
#print(successlist)
#stratcalc()

def heatPrep():
    stratx, straty = [np.linspace(0.1, 0.9, 9)] * 2
    
    xgrid, ygrid = np.meshgrid(stratx, straty)
    
    stratgrid = np.vstack(([xgrid.T], [ygrid.T])).T


def stratTally(currentRound):
    stratlist = pd.eval(gameAna[g][str(sim)].loc[currentRound, :].values)
    
    for agent in range(agentCount):
        pq = tuple(stratlist[agent][:-1])

    # idea: check for occurrence pq in stratgrid, construct an
    # nxn data-matrix consisting of counts of each strategy (or cell)
    # in stratgrid, maintain values from preceding rounds

def size_calc(currentRound):
    paylist = []
    size = 1100
    dev = 0.5
    size_map = [size] * len(agentList)
    
    for agent in Agents:
        #print("Payoff for {0}: {1}".format(agent, agent.shareData()[i][2]))
        paylist.append((agent.shareData()[currentRound][2]))

    mean = np.mean(paylist)
    stdev = np.std(paylist)
    
    for val in paylist:
        newsize = round(size + ((size*dev) * safe_div((val-mean), stdev)), 0)
        size_map[paylist.index(val)] = newsize
        
    return(size_map)



def nodeCol_calc(currentRound):
    # agents increase in colour as their distance to equal splits decreases relative to others
    color = []
    #color_map = []
    penalty = 0.8
    stratlist = pd.eval(gameAna[g][str(sim)].loc[currentRound, :].values)
    
    for strat in stratlist:
        p = strat[0]
        q = strat[1]
        # values for both p and q experience a quadratic decline as they strive further from 0.5
        p_grad = 1 - (penalty*(abs(0.5-p))**2)
        q_grad = 1 - (penalty*(abs(0.5-q))**2)
        # difference in p and q amplified
        color.append((p_grad*q_grad)**3)
        
        #cmap = plt.cm.get_cmap('RdYlGn')
        #for val in color:
        #    color_map.append(cmap(val))
        
    return(color)#_map)
        
    # agents increase in colour as their overall wallet sum increases relative to others
    """
    walletlist = []
    color_map = walletlist
    
    stratlist = []
    for agent in Agents:
        walletlist.append(np.sum(agent.wallet[:currentRound+1]))
        stratlist.append(agent.shareData()[currentRound])
    #print(stratlist)
    
    mean = np.mean(walletlist)
    stdev = np.std(walletlist)
    
    for val in color_map:
        z = round((val-mean)/stdev, 2)
        color_map[color_map.index(val)] = z #round(mean + ((mean*stdev) * ((val-mean) / stdev)), 0)
    return(color_map)
    """
    
#size_calc(5)
#nodeCol_calc(0)

#for i in range(20):
#    nodeCol_calc(i)


fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(28,13))#, 'ncols':(2)}) #plt.figure(figsize=(18,8))
# maybe change ratio for subplot columns
# make histogram depicting best agents
gs = ax1.get_gridspec()
ax1.remove()
ax3.remove()
axgraph = fig.add_subplot(gs[:2, 0])

xval = np.arange(0, rounds, 1)
offerline, = ax2.plot(xval, offerlist, color='red', label = 'average p', alpha=0.54)
acceptline, = ax2.plot(xval, acceptlist, color='midnightblue', label = 'average q', alpha=0.54)
successline, = ax2.plot(xval, successlist, color='lime', label = 'ratio successes', alpha=0.54)
lines = [offerline, acceptline, successline]

ax2.set_ylim([0, 1])
ax2.set_xticks(np.arange(0,rounds,step=5))
ax2.legend()

def animate(currentRound, xval, offerlist, acceptlist, successlist, lines):
    
    axgraph.clear()
    nx.draw(graph, pos = positions, ax=axgraph, node_color = nodeCol_calc(currentRound), alpha = 0.53, node_size = size_calc(currentRound), cmap = plt.cm.RdYlGn, width = 1.5, with_labels=True, font_size = 30)
    lines[0].set_data(xval[:currentRound], offerlist[:currentRound])
    lines[1].set_data(xval[:currentRound], acceptlist[:currentRound])
    lines[2].set_data(xval[:currentRound], successlist[:currentRound])
    return(lines,)
    # node_color=[random.choice(color_map) for j in range(len(agentList))]
    # add _r to color to get reverse colormap
    
anim = ani.FuncAnimation(fig, animate, fargs = [xval, offerlist, acceptlist, successlist, lines], frames = rounds, interval = 100, repeat_delay = 200)#, blit=True)
#plt.show()

# either use gameAna['graphtype']['sim']['agent(s)'][row:row+1] or gameAna[loc], see https://stackoverflow.com/questions/53927460/select-rows-in-pandas-multiindex-dataframe

#   to change IPython backend, enter %matplotlib followed by 'qt' for animations or 'inline' for plot pane
