# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:54:14 2020

@author: Timot
"""
import networkx as nx
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import seaborn as sns
import numpy as np
import pandas as pd
import math
import pickle
from ast import literal_eval
from matplotlib import cm
from matplotlib import colors
#from UltimatumGame import Agent, Graph
import cProfile
import re
import warnings
import subprocess
import ast
import sys

#plt.ion()
warnings.simplefilter(action='ignore', category=FutureWarning)

interpolat=False

graphType = ['Watts-Strogatz']#, 'Barabasi-Albert']

set_self = True

if set_self:
    simulations = 100#4
    rounds = 2000#2000
    agentCount = 60
    edgeDegree = 5
    
    selectionStyle = "Fermi"      # 0: unconditional, 1: proportional, 2: Fermi
    selectionIntensity = 10 # the bèta in the Fermi-equation
    
    explore = 0.00#0.005       # with prob [explore], agents adapt strategy randomly. prob [1 - explore] = unconditional/proportional imit
    

    # =============================================================================
    # GAME SETTINGS
    # =============================================================================
    
    #proportional = True
    randomRoles = False
    
    noise = True        # noise implemented as [strategy to exploit] ± noise_e 
    noise_e = 0.005
    
    updating = 1            # 0 : all agents update; 1 : at random (n) agents update
    updateN = 10
    
    testing = False
    showGraph = False

sim = 0
    
new = True
    
Gdata = {}

for selectionStyle in ['unconditional', 'proportional', 'Fermi']: #['unconditional','proportional', 'Fermi']:
   
    #for i in np.arange(0, simulations, step=20):
    for updateN in [1]:
        p_step = np.linspace(0, 0.9, simulations)
        
        #if i > 0:
        #    sys.exit()
        i = 20
        v = 1                
        g = 'Watts-Strogatz'
        
        #gameAna = pd.read_csv("Data/gameData_n{0}_sim{1}_round{2}_exp={3:.2f}_random={4}_select={5}_beta={6}_updating={7}_updateN={8}.csv".format(agentCount, simulations, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN), encoding="utf-8", header = [0,1,2])
        #edgeAna = pd.read_csv("Data/edgeData_n{0}_sim{1}_round{2}_exp={3:.2f}_random={4}_select={5}_beta={6}_updating={7}_updateN={8}.csv".format(agentCount, simulations, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN), encoding='utf-8', header = [0,1,2])
        #gameAna = pd.read_csv("Data/gameData_n{0}_sim{1}_round{2}_exp={3:.2f}_random={4}_select={5}_beta={6}_updating={7}_updateN={8}_V={9}.csv".format(agentCount, simulations, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN, i), encoding="utf-8", header = [0,1,2])
        #edgeAna = pd.read_csv("Data/edgeData_n{0}_sim{1}_round{2}_exp={3:.2f}_random={4}_select={5}_beta={6}_updating={7}_updateN={8}_V={9}.csv".format(agentCount, simulations, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN, i), encoding='utf-8', header = [0,1,2])
        gameAna = pd.read_parquet("Data/gameData_n{0}_sim{1}_round{2}_exp={3:.2f}_random={4}_select={5}_beta={6}_updating={7}_updateN={8}_V={9}.parquet".format(agentCount, simulations, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN, i), engine='pyarrow')
        edgeAna = pd.read_parquet("Data/edgeData_n{0}_sim{1}_round{2}_exp={3:.2f}_random={4}_select={5}_beta={6}_updating={7}_updateN={8}_V={9}.parquet".format(agentCount, simulations, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN, i), engine='pyarrow')
        gameAna.columns = pd.MultiIndex.from_frame(pd.DataFrame([ast.literal_eval(col) for col in gameAna.columns]))
        edgeAna.columns = pd.MultiIndex.from_frame(pd.DataFrame([ast.literal_eval(col) for col in edgeAna.columns]))
        
        
        #readFocus = open("Graphs/{0}V{1}_n{2}_sim{3}_round{4}_exp={5:.2f}_random={6}.gpickle"
        #readFocus = open("Graphs/{0}V0_n{2}_sim{3}_round{4}_exp={5:.2f}_random={6}.gpickle".format(g, sim, agentCount, simulations, rounds, explore, str(randomRoles)), 'rb') #{1}_n{2}_sim{3}_round{4}_exp={5:.2f}_prop={6}_random={7}".format(g, sim, agentCount, simulations, rounds, explore, str(randomRoles)))
        #readFocus = open("Graphs/{0}V{1}_n{2}_sim{3}_k={4}.gpickle".format(g, sim, agentCount, simulations, edgeDegree), 'rb')
        
        
        #readFocus = open("Graphs/{0}V0_n{2}_sim{3}_k={4}.gpickle".format(g, sim, agentCount, simulations, edgeDegree), 'rb')
        readFocus = open("Graphs/{0}V{1}_n{2}_sim{3}_k={4}_p={5}.gpickle".format(g, i, agentCount, simulations, edgeDegree, p_step[i]), 'rb')
        
        graph = nx.read_gpickle(readFocus)
        positions = nx.spring_layout(graph)
        
        print("data loaded")
        
        # =============================================================================
        # # Unpacking and Interpolation functions
        # =============================================================================
        
        #p_total, q_total, u_total = np.array([[[ast.literal_eval(values) for values in rnd] for rnd in np.array(gameAna[g][str(simul)])] for simul in range(simulations)]).T
        # --> returns separate p values per agent for both sims! matrix (5, 10, 2). so can use to fill p_crosssim, q_..., u_...!
        
        #p_mean, q_mean, u_mean = np.array([[[ast.literal_eval(values) for values in rnd] for rnd in np.array(gameAna[g][str(simul)])] for simul in range(simulations)]).mean(axis=0).T
        # --> yields (nxm) matrices, n=agentcount and m=rounds, with cell vals averaged over sims
        
        print("GameAna shape: {0}".format(np.shape(gameAna)))
        print("EdgeAna shape: {0}".format(np.shape(edgeAna)))
        
        """
        if new:
            if interpolat == True:
                gameData = np.array([[np.array(rnd).reshape(agentCount,3) for rnd in np.array(gameAna[g][simul])] for simul in range(simulations)]).mean(axis=0)
                edgeData = np.array([[np.array(rnd).reshape(len(graph.edges), 2, 3) for rnd in np.array(edgeAna[g][simul])] for simul in range(simulations)]).mean(axis=0)
            else:
                gameData = np.array([[[np.array(rnd).reshape(agentCount,3)] for rnd in np.array(gameAna[g][sim])] for simul in range(simulations)])
                edgeData = np.array([[[np.array(rnd).reshape(len(graph.edges), 2, 3)] for rnd in np.array(edgeAna[g][simul])] for simul in range(simulations)])[sim]
        else:
            if interpolat == True:
                gameData = np.array([[[ast.literal_eval(values) for values in rnd] for rnd in np.array(gameAna[g][str(simul)])] for simul in range(simulations)]).mean(axis=0)
                edgeData = np.array([[[ast.literal_eval(edges) for edges in rnd] for rnd in np.array(edgeAna[g][str(simul)])] for simul in range(simulations)]).mean(axis=0)
            else:
                gameData = np.array([[[ast.literal_eval(values) for values in rnd] for rnd in np.array(gameAna[g][str(simul)])] for simul in range(simulations)])[sim]
                edgeData = np.array([[[ast.literal_eval(edges) for edges in rnd] for rnd in np.array(edgeAna[g][str(simul)])] for simul in range(simulations)])[sim]
        """
        
        gameData = np.array([np.array(rnd).reshape(agentCount,3) for rnd in np.array(gameAna[g])])
        edgeData = np.array([np.array(rnd).reshape(len(graph.edges), 2, 3) for rnd in np.array(edgeAna[g])])
        
        
        
        *stratlist, u_list = gameData.T
        p_list, q_list = stratlist
        
        print('bla')
        Gdata[selectionStyle] = ['p = {0:.4f} psd = {1:.4f}, q = {2:.4f}, qsd = {3:.4f}'.format(p_list.T[-1].mean(axis=0), p_list.T[-1].std(axis=0), q_list.T[-1].mean(axis=0), q_list.T[-1].std(axis=0))]
        
        # =============================================================================
        # # Metric functions
        # =============================================================================
        
        
        """
        def safe_div(x, y):
            return 0 if y == 0 else x / y
        
        
        def stratcalc():
            avgOffer, avgAccept, avgSucc = ([], ) * 3
            
            edgeCount = len(graph.edges)
            
            *_, succ = edgeData.T
            
            avgOffer = p_list.mean(axis=0)
            varOffer = p_list.std(axis=0)
            avgAccept = q_list.mean(axis=0)
            varAccept = q_list.std(axis=0)
            avgSucc = succ.T.mean(axis=2).mean(axis=1)
            #varSucc = succ.T.var(axis=2).std(axis=1)
            
            if (avgOffer > 1).any() or (avgAccept > 1).any() or (avgSucc > 1).any():
                raise ValueError("Averages incorrect")
            return(avgOffer, varOffer, avgAccept, varAccept, avgSucc)        

        
        def size_calc():
            #size_map = np.zeros(shape = (agentCount, rounds))
            size_map = []
            
            np.shape(u_list)            
            
            for rnd in range(rounds):
                size_list = []
                size = 1100
                dev = 0.5
                
                mean = np.mean(u_list.T[rnd])
                stdev = np.std(u_list.T[rnd])
                
                #for val in u_list.T[rnd]:
                #    print(val)
                #    print("ok heyo")
                #    newsize = np.around(size + ((size*dev) * safe_div((val-mean), stdev)), 0)
                #    size_list.append(newsize)
                    #size_map[paylist.index(val), rnd] = newsize
                newsize = np.around(size + ((size*dev) * safe_div((u_list.T[rnd] - mean), stdev)), 0)
                size_list.append(newsize)
                size_map.append(size_list)
            return(size_map)
                
        
        def nodeCol_calc():
            # agents increase in colour as their distance to equal splits decreases relative to others
            cmap = cm.get_cmap('RdYlGn')            
            norm = colors.Normalize(vmin=0, vmax=1)
            
            color = [[cmap(norm(p)) for p in p_list.T[rnd]] for rnd in range(rounds)] 
            
            return(color)
        
        
        def edgeCol_calc():
            edgecol = []
            edgewidth = []
            
            for currentRound in range(rounds):
                edgetemp = []
                coltemp = []
                
                for edge in edgeData[currentRound]:                    
                    
                    coltemp.append(sum(edge.T[-1]))

                    if sum(edge.T[-1]) == 2:
                        edgetemp.append(4)
                    elif sum(edge.T[-1]) == 1:
                        edgetemp.append(2.7)
                    else:
                        edgetemp.append(1.5)  
                
                edgewidth.append(edgetemp)   
                if edgecol:
                    edgecol.append(np.add(edgecol[-1], coltemp))
                else:
                    edgecol.append(coltemp)
                     
            return(edgecol, edgewidth)    
        
        # =============================================================================
        # # Metric Creation        
        # =============================================================================
        
        offerlist, offervar, acceptlist, acceptvar, successlist = stratcalc()
        sizes = size_calc()
        col_array = nodeCol_calc()
        edgecouleurs, edgewidths = edgeCol_calc()
        
        def update(currentRound):
            nodesizes = sizes[currentRound]
            nodecolors = col_array[currentRound]
            edgecolors = edgecouleurs[currentRound]
            edgesize = edgewidths[currentRound]
            agentStrategies = np.array(stratlist).T[currentRound].T
            return(nodesizes, nodecolors, edgesize, edgecolors, agentStrategies)
        
    
        #%%
        # =============================================================================
        # # Animation Functions     
        # =============================================================================
        
        print("starting on image {0}".format(i))
        
        def run_animation():
            gs_kw = dict(width_ratios=[4,2,2], height_ratios=[1.5,2.5])
            fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2, 3, figsize=(32,13), gridspec_kw = gs_kw)#, 'ncols':(2)}) #plt.figure(figsize=(18,8))
            
            gs = ax1.get_gridspec()
            ax1.remove()
            ax4.remove()
            axgraph = fig.add_subplot(gs[:2, 0])
            fig.colorbar(cm.ScalarMappable(cmap=cm.get_cmap('RdYlGn')), ax=axgraph)
            
            gs2 = ax2.get_gridspec()
            ax2.remove()
            ax3.remove()
            axplot = fig.add_subplot(gs2[0, 1:3])
            
            xval = np.arange(0, rounds, 1)
            
            # used to be ax2
            axplot.set_ylim([0, 1])
            axplot.set_xlim([0, rounds-1])
            axplot.set_xticks(np.append(np.arange(0, rounds, step=math.floor(rounds/10)), rounds-1))
            fairline = axplot.axhline(0.5, color="black", ls='--', alpha=0.4)
            axplot.yaxis.grid()
    
            offerline, = axplot.plot([], [], lw=1, color='red', label = 'average p', alpha=0.54)
            acceptline, = axplot.plot([], [], lw=1, color='midnightblue', label = 'average q', alpha=0.54)
            successline, = axplot.plot([], [], lw=1, color='lime', label = 'ratio successes', alpha=0.54)
            lines = [offerline, acceptline, successline]#fakeline, offerline, acceptline, successline]
            axplot.legend()
            
            X = np.random.randn(6)
            Y = np.random.randn(6)
            initp, initq = np.array(stratlist).T[0].T
            
            nbins = np.linspace(0.0, 1.0, 21)
            dat, p, q = np.histogram2d(initq, initp, bins=nbins, density=False)
            ext = [q[0], q[-1], p[0], p[-1]]
            
            im = ax6.imshow(dat.T, origin='lower', cmap = plt.cm.viridis, interpolation = 'bicubic', extent = ext)#hist2d([], [], bins=20, cmap=plt.cm.BuGn)
            ax6.set_xlabel("accepts (q)")
            ax6.set_ylabel("offers (p)")
            fig.colorbar(im, ax=ax6, shrink=0.9)
            
            sns.kdeplot(initp, ax=ax5, color='r', bw=0.1, clip=(0, 1), label="p")
            sns.kdeplot(initq, ax=ax5, color='b', bw=0.1, clip=(0, 1), label="q")
            ax5.set_xlim(left = 0, right = 1) 
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
                # used to be ax2
                lines[0].set_data([],[])
                lines[1].set_data([],[])
                lines[2].set_data([],[])        
                return lines,
        
            
            def animate(currentRound):
                #since edge interaction 2x per round, animate edge round 2x currentRound? (set frames = rounds times two, maintain currentRound by (frames/2))
                
                nodesiz, nodecol, edgesize, edgecol, pqlist = update(currentRound)
                agentp, agentq = pqlist 
                
                axgraph.clear()
                ax5.clear()
                nx.draw_networkx(graph, pos = positions, ax=axgraph, edge_color = edgecol, edge_cmap = plt.cm.coolwarm, node_color = nodecol, edge_vmin=0, edge_vmax=(2*rounds), alpha = 0.53, node_size = 1200, width = edgesize, with_labels=True, font_size = 30)                
                #axgraph.table(np.around(pqlist, 2), colLabels = agentNames, cellLoc = 'center')#, cellColours = list(nodecol), cmap=plt.cm.RdYlGn)
                
                lines[0].set_data(xval[:currentRound+1], offerlist[:currentRound+1])
                lines[1].set_data(xval[:currentRound+1], acceptlist[:currentRound+1])
                lines[2].set_data(xval[:currentRound+1], successlist[:currentRound+1])
                
                pqdata, pstr, qstr = np.histogram2d(agentq, agentp, bins=nbins)
                sns.kdeplot(agentp, ax=ax5, shade=True, color='r', bw=0.1, clip=(0, 1), label="p")
                sns.kdeplot(agentq, ax=ax5, shade=True, color='b', bw=0.1, clip=(0, 1), label="q")
                im.set_data(pqdata.T)            
            
            fig.canvas.mpl_connect('button_press_event', onClick)
        
            anim = ani.FuncAnimation(fig, animate, init_func = init, frames = rounds, interval = 100, repeat_delay = 20)        
        
        
        def save_animation():
            gs_kw = dict(width_ratios=[4,2,2], height_ratios=[1.5,2.5])
            fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2, 3, figsize=(32,13), gridspec_kw = gs_kw)#, 'ncols':(2)}) #plt.figure(figsize=(18,8))
            
            gs = ax1.get_gridspec()
            ax1.remove()
            ax4.remove()
            axgraph = fig.add_subplot(gs[:2, 0])
            fig.colorbar(cm.ScalarMappable(cmap=cm.get_cmap('RdYlGn')), ax=axgraph)
            
            gs2 = ax2.get_gridspec()
            ax2.remove()
            ax3.remove()
            axplot = fig.add_subplot(gs[0, 1:3])
            
            xval = np.arange(0, rounds, 1)
            
            # used to be ax2
            axplot.set_ylim([0, 1])
            axplot.set_xlim([0, rounds-1])
            axplot.set_xticks(np.append(np.arange(0, rounds, step=math.floor(rounds/10)), rounds-1))
            fairline = axplot.axhline(0.5, color="black", ls='--', alpha=0.4)
            axplot.yaxis.grid()
    
            offerline, = axplot.plot([], [], lw=1, color='red', label = 'average p', alpha=0.54)
            acceptline, = axplot.plot([], [], lw=1, color='midnightblue', label = 'average q', alpha=0.54)
            successline, = axplot.plot([], [], lw=1, color='lime', label = 'ratio successes', alpha=0.54)
            lines = [offerline, acceptline, successline]#fakeline, offerline, acceptline, successline]
            axplot.legend()
            
            X = np.random.randn(6)
            Y = np.random.randn(6)
            initp, initq = np.array(stratlist).T[0].T
            
            nbins = np.linspace(0.0, 1.0, 21)
            dat, p, q = np.histogram2d(initq, initp, bins=nbins, density=False)
            ext = [q[0], q[-1], p[0], p[-1]]
            
            im = ax6.imshow(dat.T, origin='lower', cmap = plt.cm.viridis, interpolation = 'bicubic', extent = ext)#hist2d([], [], bins=20, cmap=plt.cm.BuGn)
            ax6.set_xlabel("accepts (q)")
            ax6.set_ylabel("offers (p)")
            fig.colorbar(im, ax=ax6, shrink=0.9)
            
            sns.kdeplot(initp, ax=ax5, color='r', bw=0.1, clip=(0, 1), label="p")
            sns.kdeplot(initq, ax=ax5, color='b', bw=0.1, clip=(0, 1), label="q")
            ax5.set_xlim(left = 0, right = 1) 
            
            canvas_width, canvas_height = fig.canvas.get_width_height()
            
            # Open an ffmpeg process
            outf = "Videos/{0}V{1}_n{2}_sim{3}_round{4}_exp={5:.2f}_random={6}_select={7}_beta={8}_updating={9}_updateN={10}.mp4".format(g, sim, agentCount, simulations, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN)
            cmdstring = ('ffmpeg', 
                     '-y', '-r', '11', # overwrite, 11fps
                     '-s', '%dx%d' % (canvas_width, canvas_height), # size of image string
                     '-pix_fmt', 'argb', # format
                     '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
                     '-vcodec', 'mpeg4', '-b:v', '5500k', outf) # output encoding
            pipp = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)
        
            def aniupdate(currentRound):
                #since edge interaction 2x per round, animate edge round 2x currentRound? (set frames = rounds times two, maintain currentRound by (frames/2))
                nodesiz, nodecol, edgesize, edgecol, pqlist = update(currentRound)
                agentp, agentq = pqlist
                
                axgraph.clear()
                ax5.clear()
        
                nx.draw_networkx(graph, pos = positions, ax=axgraph, edge_color = edgecol, edge_cmap = plt.cm.coolwarm, node_color = nodecol, edge_vmin=0, edge_vmax=(2*rounds), alpha = 0.53, node_size = 1200, width = edgesize, with_labels=True, font_size = 30)
                
                #axgraph.table(np.around(pqlist, 2), colLabels = agentNames, cellLoc = 'center')#, cellColours = list(nodecol), cmap=plt.cm.RdYlGn)
                lines[0].set_data(xval[:currentRound+1], offerlist[:currentRound+1])
                lines[1].set_data(xval[:currentRound+1], acceptlist[:currentRound+1])
                lines[2].set_data(xval[:currentRound+1], successlist[:currentRound+1])
                
                pqdata, pstr, qstr = np.histogram2d(agentq, agentp, bins=nbins)
                sns.kdeplot(agentp, ax=ax5, shade=True, color='r', bw=0.1, clip=(0, 1), label="p")
                sns.kdeplot(agentq, ax=ax5, shade=True, color='b', bw=0.1, clip=(0, 1), label="q")
                
                im.set_data(pqdata.T)
                
            for frame in range(rounds):
                if (frame+1) % 50 == 0:
                    print(frame)
                aniupdate(frame)
                fig.canvas.draw()
                
                # extract the image as an ARGB string
                pippString = fig.canvas.tostring_argb()
                # write to pipe
                pipp.stdin.write(pippString)
            
                # extract the image as an ARGB string
                pippString = fig.canvas.tostring_argb()
                # write to pipe
                pipp.stdin.write(pippString)
        
            pipp.communicate()
        
        
        def save_image():
            nodesiz, nodecol, edgesize, edgecol, pqlist = update(rounds-1)
            gs_kw = dict(width_ratios=[2,1,1])#, height_ratios=[1])
            fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2, 3, figsize=(32,13), gridspec_kw = gs_kw)#, 'ncols':(2)}) #plt.figure(figsize=(18,8))
        
            gs = ax1.get_gridspec()
            ax1.remove()
            ax4.remove()
            axgraph = fig.add_subplot(gs[:2, 0])
            fig.colorbar(cm.ScalarMappable(cmap=cm.get_cmap('RdYlGn')), ax=axgraph)
            
            gs2 = ax2.get_gridspec()
            ax2.remove()
            ax3.remove()
            axplot = fig.add_subplot(gs[0, 1:3])
            
            xval = np.arange(0, rounds, 1)
            
            nx.draw_networkx(graph, pos = positions, ax=axgraph, edge_color = edgecol, edge_cmap = plt.cm.coolwarm, node_color = nodecol, edge_vmin=0, edge_vmax=(2*rounds), alpha = 0.53, node_size = 1200, width = edgesize, with_labels=True, font_size = 30)
            
            # used to be ax2
            axplot.set_ylim([0, 1])
            axplot.set_xlim([0, rounds-1])
            axplot.set_xticks(np.append(np.arange(0, rounds, step=math.floor(rounds/20)), rounds-1))
            fairline = axplot.axhline(0.5, color="black", ls='--', alpha=0.4)
            axplot.yaxis.grid()
            
            offerline, = axplot.plot(xval, offerlist, lw=1, color='red', label = 'average p', alpha=0.8)
            acceptline, = axplot.plot(xval, acceptlist, lw=1, color='midnightblue', label = 'average q', alpha=0.8)
            successline, = axplot.plot(xval, successlist, lw=1, color='lime', label = 'ratio successes', alpha=0.8)
            axplot.fill_between(xval, offerlist - offervar, offerlist + offervar, alpha=0.3, color='red')
            axplot.fill_between(xval, acceptlist - acceptvar, acceptlist + acceptvar, alpha=0.3, color='midnightblue')
            axplot.legend()
            
            # hist & hist2d prep
            finalp, finalq = pqlist
            nbins = np.linspace(0.0, 1.0, 21)
            dat, p, q = np.histogram2d(finalq, finalp, bins=nbins, density=False)
            ext = [q[0], q[-1], p[0], p[-1]]
            
            # heatmap(hist2d)
            im = ax6.imshow(dat.T, origin='lower', cmap = plt.cm.viridis, interpolation = 'spline36', extent = ext)#hist2d([], [], bins=20, cmap=plt.cm.BuGn)
            ax6.set_xlabel("accepts (q)")
            ax6.set_ylabel("offers (p)")
            fig.colorbar(im, ax=ax6, shrink=0.9)
            
            # hist
            n, bins, patches = ax5.hist(finalp, nbins, density=1, facecolor='red', alpha=0.5, label='offer(p)')
            n, bins, patches = ax5.hist(finalq, nbins, density=1, facecolor='midnightblue', alpha=0.5, label = 'accept(q)')
            y1 = norm.pdf(nbins, finalp.mean(axis=0), finalp.std(axis=0))
            y2 = norm.pdf(nbins, finalq.mean(axis=0), finalq.std(axis=0))
            ax5.plot(nbins, y1, 'red', '--')
            ax5.plot(nbins, y2, 'midnightblue', '--')
            ax5.legend(loc='upper right')
            
            
            #sns.kdeplot(finalp, ax=ax5, color='r', bw=0.1, clip=(0, 1), label="p")
            #sns.kdeplot(finalq, ax=ax5, color='b', bw=0.1, clip=(0, 1), label="q")
            #ax5.set_xlim(left = 0, right = 1)                 
            
            plt.savefig("Images/{0}V{1}_n{2}_sim{3}_round{4}_exp={5:.2f}_random={6}_select={7}_beta={8}_updating={9}_updateN={10}_V={11}.png".format(g, sim, agentCount, simulations, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN, i))
            plt.close(fig)
        
        #run_animation()
        #save_animation()
        save_image()
       
    # %%
    # either use gameAna['graphtype']['sim']['agent(s)'][row:row+1] or gameAna[loc], see https://stackoverflow.com/questions/53927460/select-rows-in-pandas-multiindex-dataframe
    
    #   to change IPython backend, enter %matplotlib followed by 'qt' for animations or 'inline' for plot pane
    
    # agents increase in colour as their overall wallet sum increases relative to others


        gameData = np.array([[[np.array(rnd).reshape(agentCount,3)] for rnd in np.array(gameAna[g][sim])] for simul in range(simulations)])
        edgeData = np.array([[[np.array(rnd).reshape(len(graph.edges), 2, 3)] for rnd in np.array(edgeAna[g][simul])] for simul in range(simulations)])[sim]
        
        *stratlist, u_list = gameData.T
        p_list, q_list = stratlist
        
        
        # =============================================================================
        # # Metric functions
        # =============================================================================
        
        
        def safe_div(x, y):
            return 0 if y == 0 else x / y
        
        
        def stratcalc():
            avgOffer, avgAccept, avgSucc = ([], ) * 3
            
            edgeCount = len(graph.edges)
            
            *_, succ = edgeData.T
            
            avgOffer = p_list.mean(axis=0)
            varOffer = p_list.std(axis=0)
            avgAccept = q_list.mean(axis=0)
            varAccept = q_list.std(axis=0)
            avgSucc = succ.T.mean(axis=2).mean(axis=1)
            #varSucc = succ.T.var(axis=2).std(axis=1)
            
            if (avgOffer > 1).any() or (avgAccept > 1).any() or (avgSucc > 1).any():
                raise ValueError("Averages incorrect")
            return(avgOffer, varOffer, avgAccept, varAccept, avgSucc)        

        
        def size_calc():
            #size_map = np.zeros(shape = (agentCount, rounds))
            size_map = []
            
            np.shape(u_list)            
            
            for rnd in range(rounds):
                size_list = []
                size = 1100
                dev = 0.5
                
                mean = np.mean(u_list.T[rnd])
                stdev = np.std(u_list.T[rnd])
                
                #for val in u_list.T[rnd]:
                #    print(val)
                #    print("ok heyo")
                #    newsize = np.around(size + ((size*dev) * safe_div((val-mean), stdev)), 0)
                #    size_list.append(newsize)
                    #size_map[paylist.index(val), rnd] = newsize
                newsize = np.around(size + ((size*dev) * safe_div((u_list.T[rnd] - mean), stdev)), 0)
                size_list.append(newsize)
                size_map.append(size_list)
            return(size_map)
                
        
        def nodeCol_calc():
            # agents increase in colour as their distance to equal splits decreases relative to others
            cmap = cm.get_cmap('RdYlGn')            
            norm = colors.Normalize(vmin=0, vmax=1)
            
            color = [[cmap(norm(p)) for p in p_list.T[rnd]] for rnd in range(rounds)] 
            
            return(color)
        
        
        def edgeCol_calc():
            edgecol = []#np.zeros(shape=(len(graph.edges), rounds))
            edgewidth = []
            
            for currentRound in range(rounds):
                edgetemp = []
                coltemp = []
                
                for edge in edgeData[currentRound]:
                    coltemp.append(sum(edge.T[-1]))
                    
                    if sum(edge.T[-1]) == 2:
                        edgetemp.append(4)
                    elif sum(edge.T[-1]) == 1:
                        edgetemp.append(2.7)
                    else:
                        edgetemp.append(1.5)  
                
                edgewidth.append(edgetemp)   
                if edgecol:
                    edgecol.append(np.add(edgecol[-1], coltemp))
                else:
                    edgecol.append(coltemp)
                     
            return(edgecol, edgewidth)    
        
        # =============================================================================
        # # Metric Creation        
        # =============================================================================
        
        offerlist, offervar, acceptlist, acceptvar, successlist = stratcalc()
        sizes = size_calc()
        col_array = nodeCol_calc()
        edgecouleurs, edgewidths = edgeCol_calc()
        
        def update(currentRound):
            nodesizes = sizes[currentRound]
            nodecolors = col_array[currentRound]
            edgecolors = edgecouleurs[currentRound]
            edgesize = edgewidths[currentRound]
            agentStrategies = np.array(stratlist).T[currentRound].T
            return(nodesizes, nodecolors, edgesize, edgecolors, agentStrategies)
        
    
        #%%
        # =============================================================================
        # # Animation Functions     
        # =============================================================================
        
        
        print("starting on image")

        def save_image():
            nodesiz, nodecol, edgesize, edgecol, pqlist = update(rounds-1)
            gs_kw = dict(width_ratios=[2,1,1])#, height_ratios=[1])
            fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2, 3, figsize=(32,13), gridspec_kw = gs_kw)#, 'ncols':(2)}) #plt.figure(figsize=(18,8))
        
            gs = ax1.get_gridspec()
            ax1.remove()
            ax4.remove()
            axgraph = fig.add_subplot(gs[:2, 0])
            fig.colorbar(cm.ScalarMappable(cmap=cm.get_cmap('RdYlGn')), ax=axgraph)
            
            gs2 = ax2.get_gridspec()
            ax2.remove()
            ax3.remove()
            axplot = fig.add_subplot(gs[0, 1:3])
            
            xval = np.arange(0, rounds, 1)
            
            nx.draw_networkx(graph, pos = positions, ax=axgraph, edge_color = edgecol, edge_cmap = plt.cm.coolwarm, node_color = nodecol, edge_vmin=0, edge_vmax=(2*rounds), alpha = 0.53, node_size = 1200, width = edgesize, with_labels=True, font_size = 30)
            
            # used to be ax2
            axplot.set_ylim([0, 1])
            axplot.set_xlim([0, rounds-1])
            axplot.set_xticks(np.append(np.arange(0, rounds, step=math.floor(rounds/20)), rounds-1))
            fairline = axplot.axhline(0.5, color="black", ls='--', alpha=0.4)
            axplot.yaxis.grid()
            
            offerline, = axplot.plot(xval, offerlist, lw=1, color='red', label = 'average p', alpha=0.8)
            acceptline, = axplot.plot(xval, acceptlist, lw=1, color='midnightblue', label = 'average q', alpha=0.8)
            successline, = axplot.plot(xval, successlist, lw=1, color='lime', label = 'ratio successes', alpha=0.8)
            axplot.fill_between(xval, offerlist - offervar, offerlist + offervar, alpha=0.3, color='red')
            axplot.fill_between(xval, acceptlist - acceptvar, acceptlist + acceptvar, alpha=0.3, color='midnightblue')
            axplot.legend()
            
            # hist & hist2d prep
            finalp, finalq = pqlist
            nbins = np.linspace(0.0, 1.0, 200)
            dat, p, q = np.histogram2d(finalq, finalp, bins=nbins, density=False)
            ext = [q[0], q[-1], p[0], p[-1]]
            
            # heatmap(hist2d)
            im = ax6.imshow(dat.T, origin='lower', cmap = plt.cm.viridis, interpolation = 'spline36', extent = ext)#hist2d([], [], bins=20, cmap=plt.cm.BuGn)
            ax6.set_xlabel("accepts (q)")
            ax6.set_ylabel("offers (p)")
            fig.colorbar(im, ax=ax6, shrink=0.9)
            
            # hist
            n, bins, patches = ax5.hist(finalp, nbins, density=1, facecolor='red', alpha=0.5, label='offer(p)')
            n, bins, patches = ax5.hist(finalq, nbins, density=1, facecolor='midnightblue', alpha=0.5, label = 'accept(q)')
            y1 = norm.pdf(nbins, finalp.mean(axis=0), finalp.std(axis=0))
            y2 = norm.pdf(nbins, finalq.mean(axis=0), finalq.std(axis=0))
            ax5.plot(nbins, y1, 'red', '--')
            ax5.plot(nbins, y2, 'midnightblue', '--')
            ax5.legend(loc='upper right')
            
            
            #sns.kdeplot(finalp, ax=ax5, color='r', bw=0.1, clip=(0, 1), label="p")
            #sns.kdeplot(finalq, ax=ax5, color='b', bw=0.1, clip=(0, 1), label="q")
            #ax5.set_xlim(left = 0, right = 1)                 
            
            plt.savefig("Images/{0}V{1}_n{2}_sim{3}_round{4}_exp={5:.2f}_noise={12}_random={6}_select={7}_beta={8}_updating={9}_updateN={10}_V={11}.png".format(g, sim, agentCount, simulations, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN, i, noise_e))
            plt.close(fig)
        
        save_image()
       
    # %%
    # either use gameAna['graphtype']['sim']['agent(s)'][row:row+1] or gameAna[loc], see https://stackoverflow.com/questions/53927460/select-rows-in-pandas-multiindex-dataframe
    
    #   to change IPython backend, enter %matplotlib followed by 'qt' for animations or 'inline' for plot pane
    
    # agents increase in colour as their overall wallet sum increases relative to others
    """