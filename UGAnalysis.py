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
import math
import random
import pickle
import itertools
from itertools import chain
from collections import Counter
from ast import literal_eval
from matplotlib import cm
from matplotlib import colors
from UltimatumGame import Agent, Graph
#import UltimatumGame
import matplotlib.patches as mpatches
import cProfile
import re
#from collections import OrderedDict
#from colorspacious import cspace_converter


graphType = ['Watts-Strogatz', 'Barabasi-Albert']
agentList = []

set_self = True

if set_self:
    simulations = 4#4
    rounds = 1000#20
    agentCount = 6
    edgeDegree = 4
    # idea: noise around fermi-comp values?
    
    selectionStyle = "unconditional"      # 0: unconditional, 1: proportional, 2: Fermi
    selectionIntensity = 10 # the bèta in the Fermi-equation
    
    explore = 0.02       # with prob [explore], agents adapt strategy randomly. prob [1 - explore] = unconditional/proportional imit
    
    testCase = True
    

    # =============================================================================
    # GAME SETTINGS
    # =============================================================================
    
    #proportional = True
    randomPlay = False
    
    testing = False
    showGraph = False

if testCase:
    graphType = ['Testcase']

#else:
#    simulations, rounds, agentCount, edgeDegree, explore, proportional, randomPlay = settings
    
g = "Barabasi-Albert"
sim = 28

if testCase:
    g = 'testCase'
    sim = 1
    agentCount = 2
    edgeDegree = 1


with open("Data/{0}V{1}_n{2}_sim{3}_round{4}_exp={5:.2f}_random={6}_select={7}_beta={8}.pickle".format(g, sim, agentCount, simulations, rounds, explore, str(randomPlay), selectionStyle, selectionIntensity), 'rb') as f:
    Agents = pickle.load(f)

for agent in Agents:
    agentList.append(str(agent))
    #print(agent.shareData())
agentNames = list(map(str, agentList))


finalDat = []

if testCase:
    gameAna = pd.read_csv("Data/gameTest_n{0}_sim{1}_round{2}_exp={3:.2f}_random={4}_select={5}_beta={6}.csv".format(agentCount, simulations, rounds, explore, str(randomPlay), selectionStyle, selectionIntensity), encoding='utf-8', header = [0,1,2])
    edgeAna = pd.read_csv("Data/edgeTest_n{0}_sim{1}_round{2}_exp={3:.2f}_random={4}_select={5}_beta={6}.csv".format(agentCount, simulations, rounds, explore, str(randomPlay), selectionStyle, selectionIntensity), encoding='utf-8', header = [0,1,2])
else:
    gameAna = pd.read_csv("Data/gameData_n{0}_sim{1}_round{2}_exp={3:.2f}_random={4}_select={5}_beta={6}.csv".format(agentCount, simulations, rounds, explore, str(randomPlay), selectionStyle, selectionIntensity), encoding="utf-8", header = [0,1,2])
    edgeAna = pd.read_csv("Data/edgeData_n{0}_sim{1}_round{2}_exp={3:.2f}_random={4}_select={5}_beta={6}.csv".format(agentCount, simulations, rounds, explore, str(randomPlay), selectionStyle, selectionIntensity), encoding='utf-8', header = [0,1,2])

#print(gameAna)
#print(edgeAna)

readFocus = open("Graphs/{0}V{1}_n{2}_sim{3}_round{4}_exp={5:.2f}_random={6}.gpickle"
                 .format(g, sim, agentCount, simulations, rounds, explore, str(randomPlay)), 'rb') #{1}_n{2}_sim{3}_round{4}_exp={5:.2f}_prop={6}_random={7}".format(g, sim, agentCount, simulations, rounds, explore, str(randomPlay)))
graphFocus = nx.read_gpickle(readFocus)#, nodetype=int)



graph = graphFocus
positions = nx.spring_layout(graph)

# =============================================================================
# # Animation function
# =============================================================================


# %%

def safe_div(x, y):
    return 0 if y == 0 else x / y


def stratcalc():
    avgOffer = []#, avgAccept, avgSucc, strategies = ([], ) * 4
    avgAccept = []
    avgSucc = []
    
    edgeCount = len(graph.edges)
    
    for rnd in range(rounds):
        totSucc = []
        
        stratlist = gameAna[g][str(sim)].loc[rnd, :].apply(literal_eval)
        offers, accepts = zip(*[strat[:-1] for strat in stratlist])
    
        for i in range(2):
            totSucc.extend([dat[i][2] for dat in edgeAna[g][str(sim)].loc[rnd, :].apply(literal_eval)])
            #edgeAna[g][str(sim)].loc[rnd, :].values.apply
            # pd.eval(edgeAna[g][str(sim)].loc[0, :].values)[0,1,2] (from 1st edge 2nd list 3rd value)
        
        y1 = np.sum(offers)/agentCount # must be set of
        y2 = np.sum(accepts)/agentCount
        y3 = np.sum(totSucc)/(edgeCount*2) #every edge is played twice
        
        avgOffer.append(y1)
        avgAccept.append(y2)
        avgSucc.append(y3)


    return(avgOffer, avgAccept, avgSucc, stratlist)
    
offerlist, acceptlist, successlist, stratlisto = stratcalc()

def heatPrep():
    stratx, straty = [np.linspace(0.1, 0.9, 9)] * 2
    xgrid, ygrid = np.meshgrid(stratx, straty)
    griddy = np.vstack(([xgrid], [ygrid])).T
    dims = list(griddy.shape[:-1])
    dims.append(rounds)
    dat = np.zeros(shape = dims)
    
    return(griddy, dat)

#stratgrid, datagrid = heatPrep()

def stratTally():#currentRound):
    stratgrid, datagr = heatPrep()
    
    for currentRound in range(rounds):
        stratlist = gameAna[g][str(sim)].loc[currentRound, :].apply(literal_eval)
        strategies = []
        #if currentRound != 0:
        #    datagr[:,:,currentRound] = datagr[:,:,currentRound - 1] * 0.6
        
        for agent in range(agentCount):
            pq = tuple(stratlist[agent][:-1])
            strategies.append(pq)
            
        stratcount = Counter(chain(strategies))
        
        for i in range(stratgrid.shape[0]):
            for j in range(stratgrid.shape[1]):
                #print("stratgrid {0} check in strategies:\n{1}".format(tuple(stratgrid[i,j]), strategies))
                if tuple(np.around(stratgrid[i,j], 1)) in stratcount:
                    datagr[i, j][currentRound] += stratcount[tuple(np.around(stratgrid[i,j], 1))]
    return(datagr)


def size_calc():
    size_map = np.zeros(shape = (agentCount, rounds))
    
    for currentRound in range(rounds):
        paylist = []
        size = 1100
        dev = 0.5
        
        for agent in Agents:
            #print("Payoff for {0}: {1}".format(agent, agent.shareData()[i][2]))
            paylist.append((agent.shareData()[currentRound][2]))
            
        mean = np.mean(paylist)
        stdev = np.std(paylist)
        
        for val in paylist:
            newsize = round(size + ((size*dev) * safe_div((val-mean), stdev)), 0)
            size_map[paylist.index(val), currentRound] = newsize
            
    return(size_map)


def nodeCol_calc():
    # agents increase in colour as their distance to equal splits decreases relative to others
    
    #color = np.zeros(shape = (agentCount, rounds))
    color = []
    
    #cmap = plt.get_cmap("RdYlGn")
    #colors.Normalize(vmin=((1 - (penalty*(abs(0.5-0.1))**2))**2)**3, vmax=1)
    
    for currentRound in range(rounds):
        #penalty = 6#0.8
        stratlist = gameAna[g][str(sim)].loc[currentRound, :].apply(literal_eval)
        
        coltemp = []
        
        for strat in stratlist:
            p, q = strat[:-1]
            #q = strat[1]
            
            # values for both p and q experience a quadratic decline as they strive further from 0.5
            #p_grad = 1 - (penalty*(abs(0.5-p))**2)
            #q_grad = 1 - (penalty*(abs(0.5-q))**2)
            p_grad = abs(0.5-p)
            q_grad = abs(0.5-q)

            cmap = cm.get_cmap('RdYlGn')
            norm = colors.Normalize(vmin=0.1, vmax=0.9)
            """
            # V E R Y _ P A T C H Y 
            if agentCount == 2:
                ind = 0
                for strateg in stratlist:
                    if tuple(strateg) == tuple(strat):
                        color[ind, currentRound] = p_grad + q_grad
                    else:
                        ind += 1
            """
            
            #else:
                #print(stratlist)
                #print(strat)
                #color[stratlist.tolist().index(strat)] = p_grad + q_grad#((p_grad*q_grad)**3)
                #color[stratlist.tolist().index(strat)] = list(cmap(norm(p)))
                
            coltemp.append(list(cmap(norm(p))))
                
                #construct a color for sub-p and super-p
        color.append(coltemp)    
            
            
            #cmap = plt.cm.get_cmap('RdYlGn')
            #for val in color:
            #    color_map.append(cmap(val))
    
    return(color)#_map)

#nodeCol_calc()


def edgeCol_calc():
    edgecol = np.zeros(shape=(len(graph.edges), rounds))
    edgewidth = []
    
    for currentRound in range(rounds):
        edgeDat = edgeAna[g][str(sim)].loc[currentRound, :].apply(literal_eval)
        edgetemp = []
        
        if currentRound != 0:
            edgecol[:, currentRound] = edgecol[:, currentRound - 1]# * 0.6
            
        for edge in range(len(graph.edges)):
            edgecol[edge][currentRound] += sum([i[-1] for i in edgeDat[edge]])
            if sum([i[-1] for i in edgeDat[edge]]) == 2:
                edgetemp.append(4)
            elif sum([i[-1] for i in edgeDat[edge]]) == 1:
                edgetemp.append(2.7)
            else:
                edgetemp.append(1.5)
        edgewidth.append(edgetemp)
        
    return(edgecol, edgewidth)    

#edgeCol_calc()   


#def histoDat():

def agentStrats():
    stratarray = []
    
    for currentRound in range(rounds):
        stratz = []
        stratlist = gameAna[g][str(sim)].loc[currentRound, :].apply(literal_eval)
        
        for agent in range(agentCount):
            stratz.append(tuple(stratlist[agent][:-1]))
        stratarray.append([stratz])
        
    return(stratarray)

stratData = agentStrats()
sizes = size_calc()
col_array = nodeCol_calc()
edgecouleurs, edgewidths = edgeCol_calc()

datagrid = stratTally()
dataHeat = datagrid[:,:,0]


def update(currentRound):
    nodesizes = sizes[:, currentRound]
    nodecolors = col_array[currentRound]
    edgecolors = edgecouleurs[:, currentRound]
    edgesize = edgewidths[currentRound]
    dataH = datagrid[:,:, currentRound]
    agentStrategies = stratData[currentRound]
    
    return(nodesizes, nodecolors, edgesize, edgecolors, dataH, agentStrategies)


gs_kw = dict(width_ratios=[3,1,2], height_ratios=[1.5,2.5])
fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2, 3, figsize=(28,13), gridspec_kw = gs_kw)#, 'ncols':(2)}) #plt.figure(figsize=(18,8))
# maybe change ratio for subplot columns
# make histogram depicting best agents
gs = ax1.get_gridspec()
ax1.remove()
#ax3.remove()
ax4.remove()
axgraph = fig.add_subplot(gs[:2, 0])


"""
annot = axgraph.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(True)
"""

gs2 = ax2.get_gridspec()
ax2.remove()
ax3.remove()
axplot = fig.add_subplot(gs[0, 1:3])

xval = np.arange(0, rounds, 1)

# used to be ax2
axplot.set_ylim([0, 1])
axplot.set_xlim([0, rounds-1])
axplot.set_xticks(np.append(np.arange(0,rounds,step=math.floor(rounds/20)), rounds-1))
fairline = axplot.axhline(0.5, color="black", ls='--', alpha=0.4)
axplot.yaxis.grid()

#offerline, = axplot.plot(xval, offerlist, color='red', label = 'average p', alpha=0.54)
#acceptline, = axplot.plot(xval, acceptlist, color='midnightblue', label = 'average q', alpha=0.54)
#successline, = axplot.plot(xval, successlist, color='lime', label = 'ratio successes', alpha=0.54)
fakeline, = axplot.plot([], [])
offerline, = axplot.plot([], [], lw=1, color='red', label = 'average p', alpha=0.54)
acceptline, = axplot.plot([], [], lw=1, color='midnightblue', label = 'average q', alpha=0.54)
successline, = axplot.plot([], [], lw=1, color='lime', label = 'ratio successes', alpha=0.54)
lines = [fakeline, offerline, acceptline, successline]
axplot.legend()

#"""
# used to be ax4
im = ax6.imshow(dataHeat, origin="lower", interpolation="none", vmax=np.amax(datagrid))
ax6.set_xticklabels(list(np.around(np.linspace(0.0, 0.9, 10), 1)))
ax6.set_yticklabels(list(np.around(np.linspace(0.0, 0.9, 10), 1)))
ax6.set_xlabel("accepts (q)")
ax6.set_ylabel("offers (p)")
#gridtext = ax6.text('', ha = "center", va = "center", color='orange', alpha=0.75)
fig.colorbar(im, ax=ax6, shrink = 0.9)
#"""


#Artists = namedtuple("lines")
#artists = Artists()

#Writer = ani.writers['ffmpeg']
#writer = Writer(fps=22, metadata=dict(artist='Me'), bitrate=1800)

"""
def animate(currentRound, xval, offerlist, acceptlist, successlist, lines, im):
    
    axgraph.clear()
    nx.draw(graph, pos = positions, ax=axgraph, node_color = colors[:, currentRound], alpha = 0.53, node_size = sizes[:, currentRound], cmap = plt.cm.RdYlGn, width = 1.5, with_labels=True, font_size = 30)
    lines[0].set_data(xval[:currentRound], offerlist[:currentRound])
    lines[1].set_data(xval[:currentRound], acceptlist[:currentRound])
    lines[2].set_data(xval[:currentRound], successlist[:currentRound])
    im.set_data(heatFeed(currentRound))
    return(lines, im)

    # node_color=[random.choice(color_map) for j in range(len(agentList))]
    # add _r to color to get reverse colormap
    
anim = ani.FuncAnimation(fig, animate, fargs = [xval, offerlist, acceptlist, successlist, lines, im], frames = rounds, interval = 60, repeat_delay = 200)#, blit=True)
#plt.show()
"""

def run_animation():
    anim_running = True

    def onClick(event):
        nonlocal anim_running
        if anim_running:
            anim.event_source.stop()
            anim_running = False
        else:
            anim.event_source.start()
            anim_running = True
    
    
    def init():
        #nx.draw_networkx(graph, pos = positions, ax=axgraph, with_labels=True, font_size = 30) #, edge_color = edgecol, edge_cmap = plt.cm.coolwarm, node_color = nodecol, edge_vmin=0, edge_vmax=(2*rounds), alpha = 0.53, node_size = 1200, width = edgesize, with_labels=True, font_size = 30)
        # used to be ax2
        lines[0].set_data([],[])
        lines[1].set_data([],[])
        lines[2].set_data([],[])
        lines[3].set_data([],[])
        return lines[0],# lines[1], lines[2], #[0], lines[1], lines[2])# im


    def animate(currentRound):#, lines):#offerlist, acceptlist, successlist, lines):
        #since edge interaction 2x per round, animate edge round 2x currentRound? (set frames = rounds times two, maintain currentRound by (frames/2))
        nodesiz, nodecol, edgesize, edgecol, dataHeat, agentStrats = update(currentRound)
        #node_colnorm = colors.Normalize(vmin=0.2, vmax=1)
        #edge_colnorm = colors.Normalize(vmin=0, vmax=(2 * rounds)#np.amax(edgecouleurs))
        
        axgraph.clear()
        #edge_color = edgecol, edge_cmap = plt.cm.RdYlGn, 
        #node_size = nodesiz, 
        nx.draw_networkx(graph, pos = positions, ax=axgraph, edge_color = edgecol, edge_cmap = plt.cm.coolwarm, node_color = nodecol, edge_vmin=0, edge_vmax=(2*rounds), alpha = 0.53, node_size = 1200, width = edgesize, with_labels=True, font_size = 30)
        
        #annot.set_annotate("hello")
        axgraph.table(agentStrats, colLabels = agentNames, cellLoc = 'center')#, cellColours = list(nodecol), cmap=plt.cm.RdYlGn)
        lines[1].set_data(xval[:currentRound+1], offerlist[:currentRound+1])
        lines[2].set_data(xval[:currentRound+1], acceptlist[:currentRound+1])
        lines[3].set_data(xval[:currentRound+1], successlist[:currentRound+1])
        im.set_data(dataHeat)
        #ax6.text(heatText(dataHeat))
        
        return lines[0],# lines[1], lines[2],#lines[0], lines[1], lines[2])# annot, )
    
    # from here.....
    
    """
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == axgraph:
            print("this is event! {0}".format(event))
            cont, ind = gdraw.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    def update_annot(ind):
        pos = positions
        annot.xy = pos
        text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                               " ".join([names[n] for n in ind["ind"]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(0.4)
        
        fig.canvas.mpl_connect("motion_notify_event", hover)
        """
    # ...til here is hover
    
    fig.canvas.mpl_connect('button_press_event', onClick)
    

    anim = ani.FuncAnimation(fig, animate, init_func = init, frames = rounds, interval = 70, repeat_delay = 20, blit=True)#, cache_frame_data=False)#, blit=True)
    #anim.save("Data/{0}V{1}_n{2}_sim{3}_round{4}_exp={5:.2f}_random={6}_select={7}_beta={8}.pickle".format(g, sim, agentCount, simulations, rounds, explore, str(randomPlay), selectionStyle, selectionIntensity), writer = writer)
 
run_animation()
#cProfile.run('run_animation()')




# %%
# either use gameAna['graphtype']['sim']['agent(s)'][row:row+1] or gameAna[loc], see https://stackoverflow.com/questions/53927460/select-rows-in-pandas-multiindex-dataframe

#   to change IPython backend, enter %matplotlib followed by 'qt' for animations or 'inline' for plot pane

# agents increase in colour as their overall wallet sum increases relative to others
