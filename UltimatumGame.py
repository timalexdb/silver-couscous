# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 21:01:25 2019

@author: Timot
"""

import networkx as nx  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import itertools
import pickle
import sys
import collections
import math
import matplotlib.ticker as tck
from matplotlib import cm
from matplotlib import colors
from collections import OrderedDict, Counter
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from random import choice, sample
from time import process_time
from statistics import mean
import scipy as sc


class Agent:
    
    agentID = 0
    
    def __init__(self, graph):
        self.id = Agent.agentID
        self.name = "Agent " +str(self.id)
        Agent.agentID += 1
        self.node = graph[self.id]
        graph.nodes[self.id]['agent'] = self #agentID and nodeID correspond; agent accessible directly through node key
        self.neighbours = []
        self.strategy = self.randomise()        
        self.nextstrat = self.strategy[:]
        self.revenue = 0
        self.fitness = 0
        self.exemplar = self
        self.data = []
        self.randomList = np.random.rand(rounds)
        #self.randomgen = (np.random.rand() for i in range(rounds))
        #self.noiseset = ((np.random.rand(rounds*2) * noise_e) - alpha)
        self.noiseGen = (((np.random.rand() * noise_e) - alpha) for i in range(rounds*2))
        
    def meetNeighbours(self, graph):
# =============================================================================
#         # after all agents are placed on graph, agents store neighbouring agents.
#         # modelset = pregenerated list of random neighbour indices
# =============================================================================
        self.neighbours = list(graph.nodes[n]['agent'] for n in graph.neighbors(self.id))
        self.degree = len(self.neighbours)
        self.modelset = (int(random.random() * self.degree) for i in range(rounds))#(random.randint(0, self.degree-1) for i in range(rounds))#np.random.choice(self.neighbours, size=rounds, replace = True)#random.choices(self.neighbours, k=rounds)
        #self.model = self.neighbours[next(self.modelset)]
    
    def findModel(self):
        #self.model = self.neighbours[next(self.modelset)]
        self.model = self.neighbours[int(np.random.rand() * self.degree)]
        
    def storeMoney(self, currentRound):
# =============================================================================
#         # fitness calculated for agent based on interactions. 
# =============================================================================
        income = self.revenue #sum(self.revenue)
        #self.wallet.append(income)#self.wallet.append(round(np.sum(self.revenue), 2))
        #self.stratIncome.append(income) #self.stratIncome.append(round(np.sum(self.revenue), 2))
        self.fitness = income / (2 * self.degree)#len(self.neighbours))#np.mean(self.stratIncome) / (2* len(self.neighbours))  
        
        if dataStore == True or currentRound == rounds-1:
            self.data.append([self.strategy['offer'], self.strategy['accept'], income])
        if self.fitness > 1.0:
            raise ValueError("fitness no bueno chef, f = {0}".format(self.fitness))
                
        
    def findExemplar(self):
# =============================================================================
#         # find neighbour with highest fitness
# =============================================================================
        paydict = {neighbour : neighbour.fitness for neighbour in self.neighbours}
        #for neighbour in self.neighbours:
        #    paydict[neighbour] = neighbour.fitness
        best = max(paydict,key = paydict.get)
        
        if best.fitness > self.fitness:
            self.exemplar = best
        elif best.fitness == self.fitness:
            self.exemplar = choice([self, best])
        else:
            self.exemplar = self
        
        
    def updateStrategy(self, currentRound):
# =============================================================================
#         # used for comparison of update rules; obsolete but may be used later
# =============================================================================
        self.comparisonMethods(selectionStyle)(currentRound)
        
        
    def unconditional(self, currentRound):
        self.findExemplar()
        self.nextstrat = self.exemplar.strategy[:]
        
    
    def proportional(self, currentRound):
# =============================================================================
#         # model taken from random index (0, self.degree-1) from neighbours.
#         # probability draw taken from pre-gen random list for agent
# =============================================================================
        self.model = self.neighbours[next(self.modelset)]
        changeProb = (self.model.fitness - self.fitness)
        prob = self.randomList[currentRound]
        
        if changeProb > 1.0:    
            raise ValueError("changeprob over 1. modfit = {0}, selffit = {1}"
                             .format(self.model.fitness, self.fitness))
        
        if prob < changeProb:
            self.nextstrat = self.model.strategy[:]
            if verbose:
                print("{0} switch!".format(self))
            
            
    def fermi(self, currentRound):
# =============================================================================
#         # model taken from random index (0, self.degree-1) from neighbours.
#         # probability draw taken from pre-gen random list for agent
# =============================================================================
        self.model = self.modelgen#self.neighbours[next(self.modelset)]#self.modelset[currentRound]
# =============================================================================
#         fermiProb = 1 / (1 + np.e**(-selectionIntensity * (model.fitness - self.fitness)))
#         prob = self.randomList[currentRound]
#         #prob = next(self.randomgen)
#         
#         if prob < fermiProb:
#             self.nextstrat = model.strategy
#             
#             if verbose:
#                 print("& {0} fitSelf: {1}, {2} fitMod: {3}, fermiProb: {4}".format(self, self.fitness, model, model.fitness, fermiProb))
#                 print("round {4}: {0} changing strategy ({1}) to that of {2}: {3}".format(self, self.strategy, model, model.strategy, currentRound))
# =============================================================================
        return(self.model.fitness - self.fitness)


    def comparisonMethods(self, argument):
# =============================================================================
#         # returns function based on update rule setting
# =============================================================================
        switcher = {
            "unconditional": self.unconditional,
            "proportional": self.proportional,
            "Fermi": self.fermi
        }
        pairFunc = switcher.get(argument, lambda: "Invalid")
        return(pairFunc)
    
    
    def randomise(self):
# =============================================================================
#         strategy{}
#         strategy["offer"] = random.uniform(0, 1)#round(random.uniform(0, 1), 2)#choice(list(range(1,10,1)))/10
#         strategy["accept"] = random.uniform(0, 1)#round(random.uniform(0, 1), 2)#choice(list(range(1,10,1)))/10
#         if strategy["offer"] > 1.0 or strategy['accept'] > 1.0:
#             raise ValueError("randomise returns value over 1.0. p = {0}, q = {1}"
#                              .format(strategy["offer"], strategy["accept"]))
# =============================================================================
        strategy = [random.uniform(0, 1), random.uniform(0, 1)]
        if any(strategy) > 1 or any(strategy) < 0:
            raise ValueError("randomise returns value over 1.0. p = {0}, q = {1}".format(strategy[0], strategy[1]))
        return(strategy)


    def exploration(self):
# =============================================================================
#         # explore is pre-defined value by user. nextstrat and strategy both updated.
#         # explore not in use for now; code left in case it is needed later on
# =============================================================================
        if random.random() < explore:
            self.nextstrat = self.randomise()                    
            self.strategy = self.nextstrat[:]
    
    
    def changeStrat(self):
# =============================================================================
#         # changeStrat and strat calc separated s.t. agents compare and change concurrently
#         # noiseGen yields a random val in (-alpha,alpha) (alpha = noise_e/2)
#         # minmax ensures value is in (0, 1)
# =============================================================================
        if noise:
# =============================================================================
#             self.nextstrat["offer"] = min(max(self.nextstrat["accept"] + next(self.noiseGen), 0), 1)
#             self.nextstrat["accept"] = min(max(self.nextstrat["accept"] + next(self.noiseGen), 0), 1)
# =============================================================================
            self.nextstrat = (min(max(self.nextstrat[0] + next(self.noiseGen), 0), 1), min(max(self.nextstrat[1] + next(self.noiseGen), 0), 1))
        self.strategy = self.nextstrat[:]


    def kill(self):
        del self


class Graph:
     
    def __init__(self):
        self.agentCount = agentCount
        self.graphData = dict()
        self.char1 = []
        self.char2 = []
        self.charT = []
        self.graphs = []
        
         
    def createSFN(self, m, sfn_rate):
        SFN = self.altered_SFN(agentCount, m, sfn_rate)#nx.barabasi_albert_graph(agentCount, m) # scale-free network characterised by having vastly differing degrees (hence scale-free), small amount of large hubs
        SFNcharacteristics = Graph.graphCharacteristics(SFN)
        
        fnameSFN = "Barabasi-Albert_n{0}_sim{1}_m={2}".format(agentCount, simulations, m)
        
        #nx.write_gpickle(SFN, "Graphs/Experiment2/{0}.gpickle".format(fnameSFN))        
        self.graphData[fnameSFN] = SFNcharacteristics
        
        if showGraph:
            fnameSFN = "Barabasi-AlbertV_n{0}_sim{1}_m={2}".format(agentCount, simulations, m)
            
            #nx.draw(graph, node_color='r', with_labels=True, alpha=0.53, width=1.5)
            nx.draw_kamada_kawai(SFN, with_labels=True, edge_color = '#00a39c', node_color = '#ff6960', alpha=0.63, node_size = 200, width = 1)
            plt.title('{0} m = {3:0.3f}, APL={4:0.3f}, CC = {5:0.3f}, SP = {6:0.3f})'
                      .format('Barabàsi-Albert', i, simulations, m, gg.graphData[fnameSFN]['APL'], gg.graphData[fnameSFN]['CC'], gg.graphData[fnameSFN]['SPavg']))

            plt.show()
    
        return(SFN, self.graphData[fnameSFN])
    
    
    def createSWN(self, p):
                
        SWN = nx.connected_watts_strogatz_graph(agentCount, edgeDegree, p)
        
        SWNcharacteristics = Graph.graphCharacteristics(SWN)
        
        fnameSWN = "Watts-Strogatz_n{0}_k={2}_p={3}_APL={4}_CC={5}_SP={6}".format(agentCount, simulations, edgeDegree, p, SWNcharacteristics['APL'], SWNcharacteristics['CC'], SWNcharacteristics['SPavg'])
        
        #nx.write_gpickle(SWN, "Graphs/Experiment 1/{0}.gpickle".format(fnameSWN))        
        
        self.graphData[fnameSWN] = SWNcharacteristics
        
        if showGraph:
            
            nx.draw_kamada_kawai(SWN, with_labels=True, edge_color = '#00a39c', node_color = '#ff6960', alpha=0.63, node_size = 200, width = 1)
            plt.title('{0}, p = {1:0.3f}\nAPL={2:0.3f}, CC = {3:0.3f}, SP = {4:0.3f})'
                      .format('Watts-Strogatz', p, gg.graphData[fnameSWN]['APL'], gg.graphData[fnameSWN]['CC'], gg.graphData[fnameSWN]['SPavg']))
            plt.show()
    
        return(SWN, self.graphData[fnameSWN]) #self.graphs, self.graphData[fnameSWN])
    
    
    def pickRandom(self, seq, m, sfn_rate):
        targets = set()
        
        seq_items, seq_vals = zip(*Counter(seq).items())
        seq_vals = np.array(seq_vals, dtype=float)
        seq_vals += (seq_vals - 1) * sfn_rate
        
        while len(targets) < m:
            nodes = np.random.choice(seq_items, size=m-len(targets), replace=True, p=(seq_vals / seq_vals.sum()))
            for x in nodes:
                targets.add(x)
        return targets
    
    
    def altered_SFN(self, n, m, sfn_rate):
        """Returns a random graph according to the Barabási–Albert preferential
        attachment model.
    
        A graph of $n$ nodes is grown by attaching new nodes each with $m$
        edges that are preferentially attached to existing nodes with high degree.
    
        Parameters
        ----------
        n : int
            Number of nodes
        m : int
            Number of edges to attach from a new node to existing nodes
        seed : integer, random_state, or None (default)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.
    
        Returns
        -------
        G : Graph
    
        Raises
        ------
        NetworkXError
            If `m` does not satisfy ``1 <= m < n``.
    
        References
        ----------
        .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
           random networks", Science 286, pp 509-512, 1999.
        """
    
        if m < 1 or m >= n:
            raise nx.NetworkXError("Barabási–Albert network must have m >= 1"
                                   " and m < n, m = %d, n = %d" % (m, n))
    
        # Add m initial nodes (m0 in barabasi-speak)
        G = nx.empty_graph(m)
        # Target nodes for new edges
        targets = list(range(m))
        # List of existing nodes, with nodes repeated once for each adjacent edge
        repeated_nodes = []
        # Start adding the other n-m nodes. The first node is m.
        source = m
        while source < n:
            # Add edges to m nodes from the source.
            G.add_edges_from(zip([source] * m, targets))
            # Add one node to the list for each new edge just created.
            repeated_nodes.extend(targets)
            # And the new node "source" has m edges to add to the list.
            repeated_nodes.append(source)
            # Create probability distribution with updated repeated_nodes
#        len_select = len(repeated_nodes)
#        prob_dist = [1/len_select] * len_select
            # Now choose m unique nodes from the existing nodes
            # Pick uniformly from repeated_nodes (preferential attachment)
            targets = self.pickRandom(repeated_nodes, m, sfn_rate)
            source += 1
        return G
        
    def graphCharacteristics(g):
        APL = nx.average_shortest_path_length(g)    # average of all shortest paths between any node couple
        clust = nx.average_clustering(g)
        SPgraph, SPtotal = Graph.structuralPower(g)
        SPtotal = [round(SP, 4) for SP in SPtotal]
        
        charList = OrderedDict([('APL', APL), ('CC', clust), ('SPavg', SPgraph), ('SPnodes', SPtotal)])
        # for deg dist see https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_degree_histogram.html
        return charList

    def structuralPower(grph):
        SPtotal = list(range(agentCount))
        groups = dict()
        
        for node in grph.nodes:            
            groups[node] = set(grph.neighbors(node))
            groups[node].add(node)
            if testing:
                print("groups for node {0} ({1}): {2}".format(node, len(groups[node]), groups[node]))        
                
        count = 0
        
        #print(groups)
        for node in groups:
            SP = []
            reach = set()
            reach.update(groups[node])
            neighlist = []

            for member in groups:
                # if the neigh-node is not node itself and they have at least some overlap...
                if member is not node:
                    if len(groups[node].intersection(groups[member])) > 0 or member in groups[node]:
                        neighlist.append(member)
                        
                        # calculate the intersection and its length
                        intersect = groups[node].intersection(groups[member])
                        sectLen = len(intersect)
                        
                        # if node and neigh-node are actual neighbours...
                        if member in groups[node]:
                            # update the reach of focal node with neigh's neighbourhood
                            reach.update(groups[member])
                        
                        # count of all direct and one-away indirect neighbours divided by len group of member
                        SP.append(sectLen/len(groups[member]))
                        
                        count += 1
            reach.remove(node)
            
            if testing:    
                if node == agentCount-1:
                    print("\nthese are group members for node {0}: {1}".format(node, neighlist))
                    print("this is group for node {0}: {1}".format(node, groups[node]))
                    print("this is SP for {0}: {1}, {2}".format(node, SP, (sum(SP)/len(reach))))
                    print("length SP: {0}, length nbh: {1}, length reach: {2}".format(len(SP), len(groups[node]), len(reach)))
                    print("this is reach for node {0}: {1}\n".format(node, reach))
            SPtotal[node] = (sum(SP)/len(reach))
        
        if len(SPtotal) != agentCount:
            raise ValueError("something wrong with your SP.")
        
        return((np.mean(SPtotal), SPtotal))
    
       
class Population:
    
    def __init__(self, graph):
        
        self.agents = []
        self.graph = graph
        
    def populate(self):
        self.agents = [Agent(self.graph) for i in range(agentCount)]
# =============================================================================
#         birth = 0
#         while birth < agentCount:
#             agent = Agent(self.graph)
#             self.agents.append(agent)
#             birth += 1
# #        if logging == True:
# #            if latexTable == True:
# #                lafile.write("\t &&& \\\ \n")          
# =============================================================================
        for agent in self.agents:
            agent.meetNeighbours(self.graph)
# =============================================================================
#         plt.figure(figsize = (9,9))
#         nx.draw_kamada_kawai(self.graph, with_labels=True, edge_color = '#00a39c', node_color = '#ff6960', alpha=0.63, node_size = 300, width = 1)
#         plt.show()
# =============================================================================
            
    def killAgents(self):
        Agent.agentID = 0   
        for agent in self.agents:
            agent.kill()

           
class ultimatumGame:
    
    def __init__(self, graph):
        self.population = Population(graph)
        self.graph = graph
        self.edges = list(self.graph.edges)
        #self.edgeDict = {key : [] for key in self.edges}
        self.edgeList = []
        #self.successes = []
        
        if edgeDegree >= agentCount:
            raise ValueError("Amount of edges per node cannot be equal to or greater than amount of agents")
    
    def play(self):
# =============================================================================
#         # population is filled with agents at the begin of the game.
#         # playlist is the list of interaction couples.
#         # the Ultimatum Game is played twice s.t. all agents once serve as 'focal agent' that plays the UG with its neighbours.
# =============================================================================
        self.population.populate()
        self.agents = np.array(self.population.agents)
        n_edges = len(self.edges)
        
        playList = [[self.agents[node] for node in edge] for edge in self.edges]
                
        for n in range(rounds):
            
# =============================================================================
#             if randomRoles:                            
#                 randlist = np.random.rand(n_edges) * 2
# =============================================================================
                       
            for players in playList:
                
                
                strats = [agent.strategy for agent in players]                
                
                if strats[0][0] > 1.0 or strats[0][1] > 1.0 or strats[1][0] > 1.0 or strats[1][1] > 1.0:
                    raise ValueError("Values exceeding 1: p = {0}, q = {1}".format([strat[0] for strat in strats], [strat[1] for strat in strats]))
                
                if randomRoles:
                    sets = [int(np.random.rand() * 2) for i in range(2)]
                else:
                    sets = range(2)
                
                for i in sets:
                    offer = strats[i][0]
                    accept = strats[i-1][1]
                    
                    if offer >= accept:
                        players[i].revenue += (1 - offer)
                        players[i-1].revenue += offer

# =============================================================================
#                 if randomRoles:
#                     role = int(randlist[i])
#                 else:
#                     role = n % 2
#                 
#                 proposer = players[role]
#                 responder = players[role-1]
#                 
#                 offer = proposer.strategy[0]
#                 accept = responder.strategy[1]
#         
#                 if offer > 1.0 or accept > 1.0:
#                    raise ValueError("Values exceeding 1: p = {0}, q = {1}".format(offer, accept))
#                 
#                 if offer >= accept:
#                     success = 1
#                     proposer.revenue += (1 - offer)
#                     responder.revenue += offer
# =============================================================================


            for agent in self.agents:
# =============================================================================
#                 # agents calculate income and fitness; faster than agent.storeMoney()
#                 # if dataStore == False, agents only share stats at end of game
# =============================================================================
                income = agent.revenue #sum(self.revenue)
                agent.fitness = income / (2 * agent.degree)
                agent.findModel()
                
                if agent.fitness > 1.0:
                    raise ValueError("fitness no bueno chef, f = {0}".format(agent.fitness))
                    
                if dataStore == True or n == rounds-1:
                    agent.data.append([*agent.strategy, income])
            
            if n != (rounds - 1):
                self.updateAgents(n)

# =============================================================================
#                 if randomRoles:
#                     role = int(randlist[i])
#                 else:
#                     role = n % 2
#                 
#                 proposer = players[role]
#                 responder = players[role-1]
#                 
#                 offer1 = proposer.strategy[0]
#                 accept1 = responder.strategy[1]
#         
#                 if offer > 1.0 or accept > 1.0:
#                    raise ValueError("Values exceeding 1: p = {0}, q = {1}".format(offer, accept))
#                 
#                 if offer >= accept:
#                     success = 1
#                     proposer.revenue += (1 - offer)
#                     responder.revenue += offer
# =============================================================================
                
                    #if dataStore == True:
                    #self.edgeDict[self.edges[i]].append([offer, accept, success])
                    #self.edgeList.append([offer, accept, success])
                
# =============================================================================
#         # ====  E N D   O F   R O U N D S  =====
# =============================================================================
        
        if dataStore == False:
            agentdata = [[agent.data[-1]] for agent in self.agents]#self.population.agents]
            #agentdata = [[*agent.strategy, agent.income] for agent in self.agents]
            edgedata =  np.zeros((1,len(self.edges), 3), dtype='float32')
        else:
            agentdata = [agent.data for agent in self.agents]#self.population.agents]
            #edgedata = [[values[i:i+2] for values in self.edgeDict.values()] for i in range(0, rounds)]
            edgedata = np.zeros((1,len(self.edges), 3), dtype='float32')
        return(agentdata, edgedata)
    
    
    def updateAgents(self, n):
# =============================================================================
#         # updating done in steps s.t. agents update concurrently
# =============================================================================
        if updating == 1:
            agentPoule = sample(self.agents, k=updateN)
            if verbose:
                print("Agent(s) for updating: {0}".format(agentPoule))
        else:
            agentPoule = self.agents
        
        
        #for agent in agentPoule:
            #agent.updateStrategy(n)
            
            #agent.fermi(n)
            
# =============================================================================
#             #                 I N     P R O G R E S S
#             # fermi returns agent model.fitness - agent.fitness. store these in np.array
#             # then calculate fermiprob for all values in array (vectorised)
#             # then create array of rand vals and take all values in fermi-array sub prob
#             # then mask with agentPoule and take only those agents for which value over 0
#             # then update those agents' nextstrategy with modelstrategy
# =============================================================================
            
        #fermiValues = 1 / (1 + np.e**(-selectionIntensity * np.array([agent.fermi(n) for agent in agentPoule])))
        if selectionStyle == "Fermi":
            changeProbs = 1 / (1 + np.e**(-selectionIntensity * np.array([agent.model.fitness - agent.fitness for agent in agentPoule])))
        if selectionStyle == "proportional":
            changeProbs = np.array([agent.model.fitness - agent.fitness for agent in agentPoule])
            changeProbs[changeProbs < 0] = 0
        
        updateList = agentPoule[np.random.rand(len(agentPoule)) < changeProbs]
        
        for agent in updateList:
            agent.nextstrat = agent.model.strategy[:]
        
# =============================================================================
#         for agent in updateList:
#             #if agent.strategy != agent.nextstrat:
#             agent.changeStrat()
# =============================================================================
        
        if noise:
            if len(updateList) != 0:
                length_upd = len(updateList)
                stratvals = np.array([agent.nextstrat for agent in updateList]) + ((np.random.rand(length_upd, 2) * noise_e) - alpha)
                #q_array = np.array([agent.nextstrat[1] for agent in updateList]) + ((np.random.rand(length_upd) * noise_e) - alpha)
    
                #p_array = np.array([agent.nextstrat["offer"] + ((np.random.rand() * noise_e) - alpha) for agent in a])
                #q_array = np.array([agent.nextstrat["accept"] + ((np.random.rand() * noise_e) - alpha) for agent in a])
    
                stratvals[stratvals < 0] = 0
                stratvals[stratvals > 1] = 1
    
                #q_array[q_array < 0] = 0
                #q_array[q_array > 1] = 1
                #stratvals = list(stratvals)
                
                #print(p_array, q_array)
    
                for i, agent in enumerate(updateList):
                    agent.nextstrat = [*stratvals[i]]#stratvals[i]
                    agent.strategy = agent.nextstrat[:]
        else:
            agent.strategy = agent.nextstrat[:]
        
        
        for agent in self.agents[np.random.rand(agentCount) < explore]:    
            agent.nextstrat = agent.randomise()                   
            agent.strategy = agent.nextstrat[:]
            
        for agent in self.agents:
            agent.revenue = 0

    
class Plot:

    def __init__(self):
        self.name = "None"
               
        
class Simulation:
    
    def __init__(self):
        self.name = "None"
    
    def run(self, simgraphList):       
# =============================================================================
#         # simgraphList contains the graphs made for the current set of simulations.
#         # for each simulation, different graphs are created with the same p-value for each graph (for the simulation set).
#         # if dataStore == False, only last-round agent data is stored.
#         # the means for gameTest and edgeTest are calculated over the simulations.
# =============================================================================
        n_edges = len(edgeList)
        
        if dataStore == False:
            gameTest = np.empty((agentCount, 1, 3, simulations), dtype = 'float32')
            edgeTest = np.empty((1, n_edges, 3, simulations), dtype='float32')
        else:
            gameTest = np.empty((agentCount, rounds, 3, simulations), dtype='float32')
            #edgeTest = np.empty((rounds, n_edges, 3, simulations), dtype='float32')
            edgeTest = np.empty((1, n_edges, 3, simulations), dtype='float32')
        
        def playUG(simgraph):
            print('\n=== simulation {0} ==='.format(sim))
            UG = ultimatumGame(simgraph)
            gameTest[:, :, :, sim], edgeTest[:, :, :, sim] = UG.play()
            UG.population.killAgents()
                                    
        for sim in range(simulations):
            playUG(simgraphList[sim])
        
        gameTest = np.transpose(gameTest, (1,3,0,2))

        storeName = "WS-p={0}".format(p)

        datastoreSims(gameTest, storeName)
        
        
        #gameTest = gameTest.mean(axis=3)
        #edgeTest = edgeTest.mean(axis=3)
        print(gameTest.shape)
        #gameTest = np.transpose(gameTest, (1, 0, 2))
        #edgeTest = np.transpose(edgeTest, (1, 0, 2))
    
        return(gameTest, edgeTest)                

#%%

def datastoreSims(inputdata, setting):
    indexGame = pd.MultiIndex.from_product((range(simulations), agentList, ['p', 'q', 'u']), names = ['simulation', 'agent', 'value'])
    inputdata = inputdata.reshape((1,-1))

    gameData = pd.DataFrame(data = inputdata, index = range(1), columns = indexGame)
    gameData.columns = gameData.columns.map(str)
    #gameData.to_parquet("Experiment1/WS_s={3}_n={0}_k={1}_p={2}_er={4}.parquet".format(agentCount, edgeDegree, p, simulations, explore))
    gameData.to_parquet("Experiment1/new/{4}_s={2}_n={0}_k={1}_er={3}.parquet".format(agentCount, edgeDegree, simulations, explore, str(setting)))
    print(setting)

def dataHandling(gameSet):#, edgeTest):
    
    ### index: sims / agents / rounds / (entries) / variables   
    ### construct MultiIndex values to save as pd DataFrame
    for i, p in enumerate(probabilities):
        gameTest = gameSet[i][0]
        #indexGame = pd.MultiIndex.from_product((grapropphType, range(simulations), agentList, ['p', 'q', 'u']), names=['Graph', 'Simulation', 'Agent', 'value']) #graphType, range(simulations), agentList), names=['Graph', 'Simulation', 'Agent'])
        #indexEdge = pd.MultiIndex.from_product((graphType, range(simulations), edgeList, ['0', '1'], ['p', 'q', 'suc']), names=['Graph', 'Simulation', 'Edge', 'interact', 'value'])
        indexGame = pd.MultiIndex.from_product((graphType, agentList, ['p', 'q', 'u']), names=['Graph', 'Agent', 'value'])
        #indexEdge = pd.MultiIndex.from_product((graphType, edgeList, ['0', '1'], ['p', 'q', 'suc']), names=['Graph', 'Edge', 'value'])
        
        #print(np.shape(gameTest))
        #gameTest = np.stack(gameTest)
        #print(np.shape(gameTest))
        #gameTest = np.concatenate(gameTest).reshape(10000,100, 3)
        gameTest = gameTest.reshape(1, -1)
        #edgeTest = edgeTest.reshape(rounds, -1)
        
        gameData = pd.DataFrame(data=gameTest, index = [1], columns = indexGame)                
       # edgeData = pd.DataFrame(data=edgeTest, index = range(rounds), columns = indexEdge)
        
        gameData.columns = gameData.columns.map(str)
        #edgeData.columns = edgeData.columns.map(str)
        
        gameData.to_parquet("Experiment1/WattsStrogatz_s={3}_n={0}_k={1}_p={2}_er={4}.parquet".format(agentCount, edgeDegree, p, simulations, explore))
        #gameData.to_parquet("Data/gameData_n{0}_sim{1}_round{2}_exp={3:.4f}_select={4}_beta={5}_updating={6}_updateN={7}.parquet".format(agentCount, simulations, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN))
        #edgeData.to_parquet("Data/edgeData_n{0}_sim{1}_round{2}_exp={3:.4f}_random={4}_select={5}_beta={6}_updating={7}_updateN={8}.parquet".format(agentCount, simulations, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN))

#%%    

def generateImg(graph, g, positions, gamedata, edgedata, i):
    
    *stratlist, u_list = gamedata.T
    p_list, q_list = stratlist
    
    # =============================================================================
    # # Metric functions
    # =============================================================================
    
    def safe_div(x, y):
        return 0 if y == 0 else x / y
    
    def stratcalc():
        avgOffer, avgAccept, avgSucc = ([], ) * 3
        edgeCount = len(graph.edges)
        *_, succ = edgedata.T
        
        avgOffer = p_list.mean(axis=0)
        avgAccept = q_list.mean(axis=0)
        varOffer = p_list.std(axis=0)
        varAccept = q_list.std(axis=0)        
        avgSucc = succ.T.mean(axis=2).mean(axis=1)
        
        if (avgOffer > 1).any() or (avgAccept > 1).any() or (avgSucc > 1).any():
            raise ValueError("Averages incorrect")
        return(avgOffer, varOffer, avgAccept, varAccept, avgSucc)        

    def size_calc():
        size_map = []
        
        for rnd in range(rounds):
            size_list = []
            size = 1100
            dev = 0.5

            mean = np.mean(u_list.T[rnd])
            stdev = np.std(u_list.T[rnd])
            
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
            for edge in edgedata[currentRound]:
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
        
        ax2.remove()
        ax3.remove()
        axplot = fig.add_subplot(gs[0, 1:3])
        xval = np.arange(0, rounds, 1)        
        nx.draw_networkx(graph, pos = positions, ax=axgraph, edge_color = edgecol, edge_cmap = plt.cm.coolwarm, node_color = nodecol, edge_vmin=0, edge_vmax=(2*rounds), alpha = 0.53, node_size = 1200, width = edgesize, with_labels=True, font_size = 30)
        
        # used to be ax2
        axplot.set_ylim([0, 1])
        axplot.set_xlim([0, rounds-1])
        axplot.set_xticks(np.append(np.arange(1, rounds, step=math.floor(rounds/20)), rounds))
        axplot.axhline(0.5, color="black", ls='--', alpha=0.4)
        axplot.yaxis.grid()
        axplot.set_xlabel('round')

        offerline, = axplot.plot(xval, offerlist, lw=1, color='red', label = 'average p', alpha=0.8)
        acceptline, = axplot.plot(xval, acceptlist, lw=1, color='midnightblue', label = 'average q', alpha=0.8)
        successline, = axplot.plot(xval, successlist, lw=2.5, color='lime', label = 'ratio successes', alpha=0.6)
        axplot.fill_between(xval, offerlist - offervar, offerlist + offervar, alpha=0.3, color='red')
        axplot.fill_between(xval, acceptlist - acceptvar, acceptlist + acceptvar, alpha=0.3, color='midnightblue')
        axplot.legend()
        
        # hist & hist2d prep
        finalp, finalq = pqlist
        heatbins = np.linspace(0.0, 1.0, 20)
        histbins = np.linspace(0.0, 1.0, 40)
        dat, p, q = np.histogram2d(finalq, finalp, bins=heatbins, density=False)
        ext = [q[0], q[-1], p[0], p[-1]]
        
        # heatmap(hist2d)
        im = ax6.imshow(dat.T, origin='lower', cmap = plt.cm.viridis, interpolation = 'spline36', extent = ext)#hist2d([], [], bins=20, cmap=plt.cm.BuGn)
        ax6.set_xlabel("accepts (q)")
        ax6.set_ylabel("offers (p)")
        fig.colorbar(im, ax=ax6, shrink=0.9)
        
        # hist
        n, bins, patches = ax5.hist(finalp, histbins, density=0, facecolor='red', alpha=0.5, label='offers (p)')
        n, bins, patches = ax5.hist(finalq, histbins, density=0, facecolor='midnightblue', alpha=0.5, label = 'accepts (q)')
        #y1 = norm.pdf(nbins, finalp.mean(axis=0), finalp.std(axis=0))
        #y2 = norm.pdf(nbins, finalq.mean(axis=0), finalq.std(axis=0))
        #ax5.plot(nbins, y1, 'red', '--', alpha = 0.7)
        #ax5.plot(nbins, y2, 'midnightblue', '--', alpha = 0.7)
        ax5.legend(loc='upper right')
        ax5.set_xlabel('value')
        ax5.set_ylabel('')
        
        #plt.show()
        plt.savefig("Images/{0}V{1}_n{2}_round{3}_exp={4:.3f}_noise={10}_random={5}_select={6}_beta={7}_updating={8}_updateN={9}.png".format(g, i, agentCount, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN, noise_e))
        #plt.savefig("HPtest/{0}.png".format(rd))
        #plt.close(fig)
    
    save_image()

#%%


def writestats(stats, varstats):
    print(stats)
    values = [[str(val) for val in stat] for stat in stats]
    varvalues = [[str(var) for var in varstat] for varstat in varstats]
    
    varvar = list(zip(values, varvalues))
    
    with open("Experiment1/WattsStrogatz_s={3}_n={0}_k={1}_p={2}_er={4}.txt".format(agentCount, edgeDegree, len(probabilities), simulations, explore), 'w+') as f:
        for i in range(len(varvar)):
            f.write("{1}: {0}\n".format(varvar[i], ['totalp', 'pvar', 'totalq', 'qvar'][i]))
        f.close()
  
def degreeplot(totaldat, degrees):
    totalp = []
    totalq = []
    totalAPL = []
    totalCC = []
    totalSP = []
    
    degreeperc = [round(d/100.0, 2) for d in degrees]
    
    for i in totaldat:
        gamedata, edgedata, *gdata = i
        
        *stratlist, u_list = gamedata.T
        p_list, q_list = stratlist
        totalp.append(np.mean(p_list))
        totalq.append(np.mean(q_list))
        totalAPL.append(gdata['APL'])
        totalCC.append(gdata['CC'])
        totalSP.append(gdata['SPavg'])
    
    b = [totalp, totalq, totalAPL, totalCC, totalSP]
    savestats = pd.DataFrame(np.array(b).T)
    savestats.to_csv("Data/degreestatsN=100.parquet", header=None, index=None)
        
    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (15,10))
    fig = plt.figure(figsize = (15,10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    ax1.set_ylim([0,0.5])
    ax1.plot(degreeperc, totalp, 'r', label = 'average p')
    ax1.plot(degreeperc, totalq, 'b', label = 'average q')
    ax1.legend()
    ax1.set_xlabel('degree (relative to total N)')
    ax1.set_ylabel('convergence values after 5000 rounds')
    vals = ax1.get_xticks()
    ax1.set_xticklabels(['{:.0f}%'.format(val*100) for val in vals])
    ax2col = 'g'
    ax3col = 'steelblue'
    ax4col = 'm'
    ax3 = ax2.twinx()
    ax4 = ax2.twinx()

    p1, = ax2.plot(degreeperc, totalAPL, marker = '^', color = ax2col,  label = 'APL')
    p2, = ax3.plot(degreeperc, totalCC, marker = '*', color = ax3col, label = 'CC')
    p3, = ax4.plot(degreeperc, totalSP, marker = '.', color = ax4col, label = 'graph SP')
    
    ax2.set_xlabel('degree relative to total N')
    ax2.set_ylabel('Average Path Length', color=ax2col)
    ax2.tick_params(axis='y', labelcolor=ax2col)
    vals2 = ax2.get_xticks()
    ax2.set_xticklabels(['{:.0f}%'.format(val*100) for val in vals2])
    
    ax3.set_ylabel('Clustering Coefficient', color=ax3col)
    ax3.tick_params(axis='y', labelcolor=ax3col)
    ax3.set_yticks([i/100 for i in range(0, 110, 20)])
    
    ax4.set_ylabel('Average Structural Power in graph', color=ax4col)
    ax4.tick_params(axis='y', labelcolor=ax4col)
    ax4.set_yticks([i/100 for i in range(0, 110, 20)])
    ax4.spines["right"].set_position(("axes", 1.057))
    
    lines = [p1, p2, p3]
    ax2.legend(lines, [l.get_label() for l in lines])

def ex2plot(totaldat, graphs):
    
    totalp = []
    totalpvar = []
    totalq = []
    totalqvar = []
    
    tot_labels = []
    
    for i in totaldat:
        gamedata, edgedata, *gdata = i
        
        *stratlist, u_list = gamedata.T

        p_list, q_list = stratlist
        totalp.append(p_list.mean(axis=0).mean(axis=0))
        totalpvar.append(p_list.mean(axis=0).var(axis=0))
        totalq.append(q_list.mean(axis=0).mean(axis=0))
        totalqvar.append(q_list.mean(axis=0).var(axis=0))

        p_labels = np.array(p_list)
        q_labels = np.array(q_list)

        p_labels[p_labels >= 0.5] = 1
        p_labels[p_labels < 0.5] = 3
        q_labels[q_labels >= 0.5] = 1
        q_labels[q_labels < 0.5] = 0
        
        tot = p_labels + q_labels
        tot_labels.append(tot)
    
    #print(tot)
    #print(np.shape(tot))        
    print(totalp, totalq)
    print(totalpvar, totalqvar)
    
    totalp = np.array(totalp)
    totalpvar = np.array(totalpvar)
    totalq = np.array(totalq)
    totalqvar = np.array(totalqvar)
    
    fig = plt.figure(figsize = (10, 10))#(15,10))
    ax1 = fig.add_subplot(211)
    #ax2 = fig.add_subplot(212)
    
    setting = ['RND', 'SFN']
    
    ax1.set_ylim([0, 0.5])
    ax1.plot(setting, totalp, 'r', label = 'average p')
    ax1.plot(setting, totalq, 'b', label = 'average q')
    ax1.fill_between(setting, totalp - totalpvar, totalp + totalpvar, alpha=0.2, color='red')
    ax1.fill_between(setting, totalq - totalqvar, totalq + totalqvar, alpha=0.2, color='blue')
    ax1.legend()
    
    ax1.set_xlabel('setting')
    ax1.set_ylabel('strategy values')
    
    for i in range(2):
        plt.figure(figsize = (9,9))
        nx.draw_kamada_kawai(graphs[i][0], with_labels=True, edge_color = '#00a39c', node_color = '#ff6960', alpha=0.63, node_size = 300, width = 1)
    
    degree_hist(graphs)
        



def ex1plot(totaldat, probs, graphs):
    
    allpq = []
    totalp = []
    totalpvar = []
    totalq = []
    totalqvar = []
    
    altruist = []
    demand = []
    sgpn = []
    exploit = []
    tot_labels = []
    
    for i in totaldat:
        gamedata, edgedata, *gdata = i
        
        *stratlist, u_list = gamedata.T
        p_list, q_list = stratlist
        allpq.append(stratlist)
        
        totalp.append(p_list.mean(axis=0).mean(axis=0))
        totalpvar.append(p_list.mean(axis=0).std(axis=0))
        totalq.append(q_list.mean(axis=0).mean(axis=0))
        totalqvar.append(q_list.mean(axis=0).std(axis=0))
        
        p_labels = np.array(p_list)
        q_labels = np.array(q_list)
        
        p_labels[p_labels >= 0.5] = 1
        p_labels[p_labels < 0.5] = 3
        q_labels[q_labels >= 0.5] = 1
        q_labels[q_labels < 0.5] = 0
        
        tot = p_labels + q_labels
        tot_labels.append(tot)
        
        
    totalp = np.array(totalp)
    totalpvar = np.array(totalpvar)
    totalq = np.array(totalq)
    totalqvar = np.array(totalqvar)
    #print(totalpvar, totalqvar)
    
    #writestats([totalp, totalq], [totalpvar, totalqvar])
    #print(totalp, totalq)
    
    tot_labels = np.asarray(tot_labels, dtype='int')
    print(tot_labels.shape)
    print("all ok")
    stratStyles = OrderedDict({'altruist' : [], 'demand' : [], 'sgpn' : [], 'exploit': []})
    for probset in tot_labels:
        simlist = []
        for sim in probset.T:
            simlist.extend(sim)
        labels = Counter(simlist)
        #lab, nrs = zip(*labels.items())
        #nrs = np.array(nrs)/simulations
        #labeldict = {k:v for k,v in zip(lab, nrs)}
        #print(labels)
        #print(labels[3])
        for i, key in enumerate(stratStyles):
            stratStyles[key].append(labels[i+1]/simulations)
    #print(stratStyles)
        # for i, key in enumerate(stratStyles):
        #     stratStyles[key].append(lab[i+1]/agentCount)    
    #b = [totalp, totalq, stratStyles['altruist'], stratStyles['demand'], stratStyles['sgpn'], stratStyles['exploit']]
    
    #savestats = pd.DataFrame(np.array(b).T)
    #savestats.to_csv("Data/degreestatsN=100.parquet", header=None, index=None)    
    
    fig = plt.figure(figsize = (8, 5))#(15,10))
    ax1 = fig.add_subplot(211)
    #ax2 = fig.add_subplot(212)
    ax_swn = ax1.twinx()
    
    ax1.set_ylim([0, 0.5])
    ax1.plot(probs, totalp, 'r', label = 'average p')
    ax1.plot(probs, totalq, 'b', label = 'average q')
    ax1.fill_between(probs, totalp - totalpvar, totalp + totalpvar, alpha=0.15, color='red')
    ax1.fill_between(probs, totalq - totalqvar, totalq + totalqvar, alpha=0.15, color='blue')
    ax1.set_xscale('symlog', linthreshx=0.00999)
    ax1.set_xticks([0, 0.01, 0.1, 1])
    ax1.set_xlim(xmin=-0.0001, xmax=1.05)
    ax1.get_xaxis().set_major_formatter(tck.ScalarFormatter())
    
    #print(totalp)
    #print(totalpvar)
    #print(totalq)
    #print(totalqvar)
    
    ax1.set_xlabel('rewiring probability')
    ax1.set_ylabel('strategy values')
    
    # S M A L L - W O R L D N E S S

    # omegaList = []
    # print(len(graphs))
    # for setting in graphs:
    #     # omegatemp = []
    #     # print(setting[:2])
    #     # for gr in setting[:1]:
    #     #     omegatemp.append(nx.omega(gr))
        
    #     #omegaList.append(np.array(omegatemp) / len(setting[:2]))
    #     gr = setting[0]
    #     omegaList.append(nx.omega(gr))
    omegaList = [-0.6103799283154121, -0.4014478150275076, -0.1500626123612917,-0.066543601507847, 0.24363851727982166, 0.38774445893089954,
                 0.5462974429235774, 0.5446757189047882, 0.6119171753025963, 0.8235778786211955, 0.9370779741235721]
    
    swn_probs = [probs[i-1] for i in range(len(probs)+1) if i % 2 != 0]
    
    ax_swn.plot(swn_probs, omegaList, color='dimgray', label = chr(969), linewidth = 0.7)
    
    ax_swn.set_ylabel('small-worldness ({0})'.format(chr(969)), color='dimgray')
    ax_swn.tick_params(axis='y', labelcolor='dimgray')
    ax_swn.set_ylim([-1, 1])
    ax_swn.axhline(y=0, linestyle = ':', color='dimgray', linewidth = 1)
    ax1.legend()

    #vals = ax1.get_xticks()
    #ax1.set_xticklabels(['{:.0f}%'.format(val*100) for val in vals])
    
    altruistC = 'forestgreen'
    demandC = 'cadetblue'
    sgpnC = 'maroon'
    exploitC = 'orange'
    
# =============================================================================
#     # T H I S  I S  F O R  L I N E P L O T
#
#     ax2.set_ylim([-0.1, 1.1])
#     
#     l1, = ax2.plot(probs, stratStyles['altruist'], color = altruistC,  label = 'altruist')
#     l2, = ax2.plot(probs, stratStyles['demand'], color = demandC, label = 'demand')
#     l3, = ax2.plot(probs, stratStyles['sgpn'], color = sgpnC, label = 'sgpn')
#     l4, = ax2.plot(probs, stratStyles['exploit'], color = exploitC, label = 'exploit')
#     
#     ax2.set_xlabel('rewiring probability')
#     ax2.set_ylabel('ratio strategy type')
#     
#     lines = [l1, l2, l3, l4]
#     ax2.legend(lines, [l.get_label() for l in lines])
# =============================================================================
    indices = []
    newprobs = [0.0, 0.085, 0.105, 0.915]
    for p in newprobs:
        indices.append(probs.index(p))
    #indices = [index for index in range(len(probs)) for p in ['0.0, 0.085, 0.105, 0.915'] if probs[index] == p]
        
    print(totalp[indices])
    print(totalpvar[indices])
    print(totalq[indices])
    print(totalqvar[indices])
    
    pqrelevant = np.array(allpq)[indices]
    
    for i, prob_pq in enumerate(pqrelevant):
        fig2 = plt.figure(figsize = (9, 4))
        ax_strats = fig2.add_subplot(121)
        axheat = fig2.add_subplot(122)
        fig2.suptitle('{1} = {0}'.format(newprobs[i], r'$p_{rewire}$'))
        # hist & hist2d prep
        phist, qhist = prob_pq
    
        phist = phist.flatten()
        qhist = qhist.flatten()
        
        heatbins = np.linspace(0.0, 1.0, 40)
        histbins = np.linspace(0.0, 1.0, 40)
        dat, p, q = np.histogram2d(qhist, phist, bins=heatbins, density=False)
        ext = [q[0], q[-1], p[0], p[-1]]
        
        # heatmap(hist2d)
        im = axheat.imshow(dat.T, origin='lower', cmap = plt.cm.viridis, interpolation = 'spline36', extent = ext)#hist2d([], [], bins=20, cmap=plt.cm.BuGn)
        axheat.set_xlabel("accepts (q)")
        axheat.set_ylabel("offers (p)")
        fig.colorbar(im, ax=axheat, shrink=0.9)
        
        # hist
        n, bins, patches = ax_strats.hist(phist, histbins, density=0, facecolor='red', alpha=0.5, label='offers (p)')
        n, bins, patches = ax_strats.hist(qhist, histbins, density=0, facecolor='midnightblue', alpha=0.5, label = 'accepts (q)')
        #y1 = sc.stats.norm.pdf(histbins, phist.mean(axis=0), phist.std(axis=0))
        #y2 = sc.stats.norm.pdf(histbins, qhist.mean(axis=0), qhist.std(axis=0))
        #ax_strats.plot(histbins, y1, 'red', '--', alpha = 0.6)
        #ax_strats.plot(histbins, y2, 'midnightblue', '--', alpha = 0.6)
        ax_strats.legend(loc='upper right')
        ax_strats.set_xlabel('strategy value')
        ax_strats.set_ylabel('frequency')
        plt.show(fig2)
        plt.close(fig2)
        


def demographCalc(totaldata, probs, stratStyles):
    
    fig = plt.figure(figsize = (10, 5))
    ax2 = fig.add_subplot(111)
    
    w=0.01
    ax2.set_xlim(0-w*3, 1)
    ax2.set_ylim(0,1)
    probs = np.array(probs)
    #print(stratStyles.())
    ax2.bar(probs-w*2, stratStyles['altruist'], width=w, color = altruistC, label = 'altruist')
    ax2.bar(probs-w, stratStyles['demand'], width=w, color = demandC, label = 'demand')
    ax2.bar(probs+w, stratStyles['sgpn'], width=w, color = sgpnC, label = 'sgpn')
    ax2.bar(probs+w*2, stratStyles['exploit'], width=w, color = exploitC, label = 'exploit')
    ax2.legend()
    #ax2.set_xscale('symlog', linthreshx=0.00999)
    #ax2.set_xticks([0, 0.01, 0.1, 1])
    #ax2.set_xlim(xmin=-0.0001, xmax=1.05)
    #ax2.get_xaxis().set_major_formatter(tck.ScalarFormatter())
    #bars = [b1, b2, b3, b4]
    #ax2.legend(bars, [b.get_label() for b in bars])
    

def degree_hist(graphset):
    degs = []
    degCounts = []
    
    for setting in graphset:
        gnum = len(setting)
        degseq = []
        
        for graph in setting:
            degseq.extend(sorted([d for n, d in graph.degree()], reverse=True))  # degree sequence
            
        degreeCount = collections.Counter(degseq)
        deg, cnt = zip(*degreeCount.items())
        cnt = np.array(cnt)/gnum
        
        degs.append(deg)
        degCounts.append(cnt)
    
    fig = plt.figure(figsize = (10, 4))#(15,10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    #axtotal = fig.add_subplot(111, frameon=False)
    #axtotal.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    
    ax1.bar(degs[0], degCounts[0], width = 0.80, color = '#00a39c')
    ax2.bar(degs[1], degCounts[1], width = 0.80, color = '#00a39c')
    ax1.set_xlabel("degree")
    ax2.set_xlabel("degree")
    ax1.set_ylabel("average frequency")
    ax2.set_ylabel("average frequency")
    ax1.set_title("random networks")
    ax2.set_title("scale-free networks")
    #ax1.set_xlabel("random network")
    #ax2.set_xlabel("scale-free network")
# =============================================================================
#     fig, ax = plt.subplots()
#     plt.bar(deg, cnt, width=0.80, color='#00a39c')
# =============================================================================
    
    #plt.ylabel("average frequency")
    #plt.xlabel("degree")
        
    plt.show()    
    

def betaplot(totaldat, betalist):
    totalp = []
    totalq = []
    
    for i in totaldat:
        print(np.shape(i))
        gamedata, edgedata, *_ = i
        
        *stratlist, u_list = gamedata.T
        p_list, q_list = stratlist
        totalp.append(np.mean(p_list))
        totalq.append(np.mean(q_list))
    
    fig = plt.figure(figsize = (15,5))
    ax1 = fig.add_subplot(111)
    
    ax1.plot(betalist, totalp, 'r', label = 'average p')
    ax1.plot(betalist, totalq, 'b', label = 'average q')
    ax1.legend()
    ax1.set_xlabel('bèta (selection intensity)')
    ax1.set_ylabel('strategy values')
    

    """
    degreeperc = [round(d/100.0, 2) for d in degrees]
    
    for i in totaldat:
        gamedata, edgedata, gdata = i
        
        *stratlist, u_list = gamedata.T
        p_list, q_list = stratlist
        totalp.append(np.mean(p_list))
        totalq.append(np.mean(q_list))
        totalAPL.append(gdata['APL'])
        totalCC.append(gdata['CC'])
        totalSP.append(gdata['SPavg'])
    
    b = [totalp, totalq, totalAPL, totalCC, totalSP]
    savestats = pd.DataFrame(np.array(b).T)
    savestats.to_csv("Data/degreestatsN=100.parquet", header=None, index=None)
    
    #print("multiplot output \n{0}\n{1}".format(totalp, gdata))
    
    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (15,10))
    fig = plt.figure(figsize = (15,10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    ax1.set_ylim([0,0.5])
    #ax1.set_xlim([degrees[0], degrees[-1]])
    ax1.plot(degreeperc, totalp, 'r', label = 'average p')
    ax1.plot(degreeperc, totalq, 'b', label = 'average q')
    ax1.legend()
    ax1.set_xlabel('degree (relative to total N)')
    ax1.set_ylabel('convergence values after 5000 rounds')
    vals = ax1.get_xticks()
    ax1.set_xticklabels(['{:.0f}%'.format(val*100) for val in vals])
    #plt.plot(totalp, 'r', totalq, 'b')
    print(degreeperc)
    ax2col = 'g'
    ax3col = 'steelblue'
    ax4col = 'm'
    ax3 = ax2.twinx()
    ax4 = ax2.twinx()

    p1, = ax2.plot(degreeperc, totalAPL, marker = '^', color = ax2col,  label = 'APL')
    p2, = ax3.plot(degreeperc, totalCC, marker = '*', color = ax3col, label = 'CC')
    p3, = ax4.plot(degreeperc, totalSP, marker = '.', color = ax4col, label = 'graph SP')
    
    #ax2.set_xlim([degrees[0], degrees[-1]])
    ax2.set_xlabel('degree relative to total N')
    ax2.set_ylabel('Average Path Length', color=ax2col)
    ax2.tick_params(axis='y', labelcolor=ax2col)
    vals2 = ax2.get_xticks()
    ax2.set_xticklabels(['{:.0f}%'.format(val*100) for val in vals2])
    
    ax3.set_ylabel('Clustering Coefficient', color=ax3col)
    ax3.tick_params(axis='y', labelcolor=ax3col)
    ax3.set_yticks([i/100 for i in range(0, 110, 20)])
    
    ax4.set_ylabel('Average Structural Power in graph', color=ax4col)
    ax4.tick_params(axis='y', labelcolor=ax4col)
    ax4.set_yticks([i/100 for i in range(0, 110, 20)])
    ax4.spines["right"].set_position(("axes", 1.057))
    
    lines = [p1, p2, p3]
    ax2.legend(lines, [l.get_label() for l in lines])
    """ 

    
def SWNplot(totgraphdata):
    
    totalAPL = []
    totalCC = []
    totalSP = []
    
    print(totgraphdata[0])
    
    for dat in totgraphdata:
        #print(dat)
        #print(dat['APL'])
        totalAPL.append(dat['APL'])
        totalCC.append(dat['CC'])
        totalSP.append(dat['SPavg'])
    
    print("length apl {0}".format(len(totalAPL)))
    
    print(min(totalCC), max(totalCC))
    
    CCstep20 = (max(totalCC)-min(totalCC))/20
    
    x_vals = np.arange(0, 1, 0.005)

    def find_nearest(arr, steps):
        cc_list = []
        prob_list = []
        
        CCstep = (max(totalCC) - min(totalCC)) / steps

        
        for i in range(steps + 1):
            value = max(totalCC) - CCstep * (i)
            idx = np.abs(np.array(arr) - value).argmin()
            
            cc_list.append(totalCC[idx])
            prob_list.append(x_vals[idx])
            
            print("CC: {0:.4f} for rewiring prob: {1:.4f}".format(cc_list[-1], prob_list[-1]))
            
        return(cc_list, prob_list)
        
    testcc, testprobs = find_nearest(totalCC, 20)
        
    print("test for these values: \n{0}\nand these\n{1}".format(testcc, testprobs))    
    
    fig = plt.figure(figsize = (15,7))
    ax1 = fig.add_subplot(111)
    
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    
    l1, = ax1.plot(x_vals, totalAPL, marker = '^', color='g', label = 'APL')
    l2, = ax2.plot(x_vals, totalCC, marker = '*', color='steelblue', label = 'CC')
    l3, = ax3.plot(x_vals, totalSP, marker = '.', color='m', label = 'SP')
    
    ax1.set_xlabel('rewiring probability')    
    ax1.set_xscale('symlog', linthreshx=0.00999)
    ax1.set_xticks([0, 0.01, 0.1, 1])
    ax1.set_xlim(xmin=-0.0001, xmax=1.05)
    ax1.get_xaxis().set_major_formatter(tck.ScalarFormatter())
    
    wherelist = []
    for xval in testprobs:
        where = np.where(x_vals == xval)[0][0]
        wherelist.append(where)
        ax1.axvline(x=xval, ymax = totalCC[where], linestyle = '-', linewidth = 0.5, color='steelblue')
    
    ax1.set_ylabel('Average Path Length', color = 'g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.set_ylim([0, 7])
    
    ax2.set_ylabel('Clustering Coefficient', color='steelblue')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2.set_ylim([0, 1])
    ax2.axhline(y=totalCC[0], linestyle = ':', color='steelblue', linewidth = 1)
    
    ax3.set_ylabel('Average Structural Power in graph', color='m')
    ax3.tick_params(axis='y', labelcolor='m')
    ax3.set_ylim([0, 1])
    ax3.spines["right"].set_position(("axes", 1.06))
    ax3.axhline(y=totalSP[0], linestyle = ':', color='m', linewidth = 1)
        
    lines = [l1, l2, l3]
    ax1.legend(lines, [l.get_label() for l in lines])
        
    #b = [totalAPL, totalCC, totalSP]
    #savestats = pd.DataFrame(np.array(b).T)
    #savestats.to_csv("Data/clusteringstatsN=100.parquet", header=None, index=None)
    pd.DataFrame(testprobs).to_csv("Data/testprobabilitiesN=100.parquet", header=None, index=None)
    pd.DataFrame(testcc).to_csv("Data/testCCN=100.parquet", header=None, index=None)


    # %%
    # either use gameAna['graphtype']['sim']['agent(s)'][row:row+1] or gameAna[loc], see https://stackoverflow.com/questions/53927460/select-rows-in-pandas-multiindex-dataframe
    
    #   to change IPython backend, enter %matplotlib followed by 'qt' for animations or 'inline' for plot pane
    
    # agents increase in colour as their overall wallet sum increases relative to others                                
                
  
if __name__ == '__main__':
    # =============================================================================
    # HYPERPARAM
    # =============================================================================
    
    simulations = 50
    rounds = 10000#10000
    agentCount = 60
    edgeDegree = 4
    
    selectionStyle = "Fermi"      # 0: unconditional, 1: proportional, 2: Fermi
    selectionIntensity = 10 # the bèta in the Fermi-equation
    
    explore = 0.001       # with prob [explore], agents adapt strategy randomly. prob [1 - explore] = unconditional/proportional imit

    # =============================================================================
    # GAME SETTINGS
    # =============================================================================
    
    randomRoles = True  # if false, focal agent is assigned role randomly
  
    noise = True        # noise implemented as [strategy to exploit] ± noise_e 
    noise_e = 0.01
    alpha = noise_e/2
    
    updating = 0            # 0 : all agents update; 1 : at random (n) agents update
    updateN = 1
    
    testing = False
    showGraph = False
    
    logging = False
    latexTable = False
    
    dataStore = False
    
    verbose = False
    
    graphType = ['Watts-Strogatz']
    
    #gg = Graph()
    #graphs, gdata = gg.createGraph()
    
    totaldata = []
    times = []
    degrees = []
    
# =============================================================================
#     totalgraphdata = []
#     realizations = 100
#     for p in np.arange(0, 1, 0.005):
#         
#         print('commence p = {0}'.format(p))
#         interim = {key: 0 for key in ('APL', 'CC', 'SPavg')}
#         for real in range(realizations):
#             _, gdata = gg.createSWN(p)
# 
#             for key, value in gdata.items():
#                 if key != 'SPnodes':
#                     interim[key] += value
#         
#         for key, value in interim.items():
#             interim[key] = (value/realizations)
#         
#         totalgraphdata.append(interim)
#     
#     SWNplot(totalgraphdata)
#     
# =============================================================================
    
#    betaList = []
    #probabilities = [0.0, 0.015, 0.035, 0.05, 0.07, 0.085, 0.105, 0.13, 0.15, 0.17, 0.2, 0.225, 0.25, 0.28500000000000003, 0.315, 0.355, 0.395, 0.45, 0.525, 0.61, 0.915]
    #probabilities = [0.0, 0.035, 0.07, 0.105, 0.15, 0.2, 0.25, 0.315, 0.395, 0.525, 0.915]
    
    #probabilities = [0.0, 0.915]
    
    gg = Graph()
    graphSet = []
    gdataSet = []
    
# =============================================================================
#     #W A T T S - S T R O G A T Z   N E T W O R K    S E T   G E N E R A T O R
#     for prob in probabilities:
#         problist = []
#         gdatproblist = []
#         for sim in range(simulations):
#             currgraph, gdata = gg.createSWN(prob)        
#             problist.append(currgraph)
#             gdatproblist.append(gdata)
#         graphSet.append(problist)
#         gdataSet.append(gdatproblist)
# =============================================================================
        
    # S F N   N E T W O R K    S E T   G E N E R A T O R
    sfn_ratelist = np.arange(0, 2.2, 0.2)
    
    for sfn_rate in sfn_ratelist:
        gtemp = []
        gdattemp = []
        for sim in range(simulations):
            currgraph, gdata = gg.createSFN(4, sfn_rate)
            
            gtemp.append(currgraph)
            gdattemp.append(gdata)
        graphSet.append(gtemp)
        gdataSet.append(gdattemp)
        
        
    for i in range(len(sfn_ratelist)):
#        p = probabilities[i]
        print("now in setting sfw_rate = {0}".format(sfn_ratelist[i]))
        
        graphList = graphSet[i]
        gdataList = gdataSet[i]
                
        edgeList = [str(edge) for edge in graphList[0].edges]
        agentList = ["Agent %d" % agent for agent in range(0, agentCount)]
        
        start = process_time()
        
# =============================================================================
#         if logging == True:
#             file=open("Logs/{0}V{1}_n{2}_round{3}_exp={4:.3f}_noise={10}_random={5}_select={6}_beta={7}_updating={8}_updateN={9}.txt".format(g, i, agentCount, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN, noise_e), 'w+')
#         if latexTable == True:
#             lafile=open("Latex/{0}V{1}_n{2}_round{3}_exp={4:.3f}_noise={10}_random={5}_select={6}_beta={7}_updating={8}_updateN={9}.txt".format(g, i, agentCount, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN, noise_e), 'w+')
# =============================================================================
        
        gamedat, edgedat = Simulation().run(graphList)
        totaldata.append((gamedat[-1], edgedat[-1], gdataList))
        
        #generateImg(graph, g, positions, gamedat, edgedat, i)
# =============================================================================
#         if logging == True:
#             file.close()
#         if latexTable == True:
#             lafile.close()
# =============================================================================
        
        stop = process_time()
        times.append(stop-start)
        
        #if dataStore == True:
        #    dataHandling(gamedat, edgedat)
    
# =============================================================================
#     edges = []
#     for graphs in graphSet:
#         edgtemp = 0
#         for graph in graphs:
#             edgtemp += len(graph.edges)
#         edges.append(edgtemp/simulations)
#     print(edges)
# =============================================================================
    
# =============================================================================
#     for i, p in enumerate(probabilities):
#         datastoreSims(totaldata[i][0], 'prob={0}'.format(p))
# =============================================================================
    
    #ex2plot(totaldata, graphSet) 
    ex1plot(totaldata, probabilities, graphSet)

    #betaplot with betas [5, 10, 15, 20, 25, 30]
    #betaplot(totaltotaldata, betaList)
    
    #degreeplot(totaldata, degrees)
    
    print("average process time: {0} s\ntotal process time: {1} s\nall process times: {2}".format(mean(times), sum(times), times))


#dataHandling(totaldata)


    
