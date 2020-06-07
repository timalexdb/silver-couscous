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
from scipy.stats import norm
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from ast import literal_eval
from random import choice, sample
from time import process_time
from statistics import mean


class Agent:
    
    agentID = 0
    
    def __init__(self, graph):
        self.id = Agent.agentID
        self.name = "Agent " +str(self.id)
        Agent.agentID += 1

        self.node = graph[self.id]
        graph.nodes[self.id]['agent'] = self #agentID and nodeID correspond; agent accessible directly through node key
        self.neighbours = []
        self.degree = 0
        
        self.strategy = self.randomise()
        #self.strategy = {}
        #self.strategy['offer'], self.strategy['accept'] = [(0.8, 0.1), (0.5, 0.1)][self.id]
        
        self.nextstrat = self.strategy
        #self.wallet = []        # wallet keeps track of total revenue
        #self.revenue = []       # revenue keeps track of #gains per round
        self.revenue = 0
        self.stratIncome = []
        self.fitness = 0
        self.exemplar = self
        self.data = []
    
    
    def __repr__(self):
        return("Agent %d" %self.id)# + "S({0},{1})".format(self.strategy['offer'], self.strategy['accept']))
    
    def meetNeighbours(self, graph):
        #print("{0} meeting neighbours! : {1}".format(self, self.neighbours))
        self.neighbours = list(graph.nodes[n]['agent'] for n in graph.neighbors(self.id))
        self.degree = len(self.neighbours)
        #print("{0} met neighbours! : {1}".format(self, self.neighbours))
    
    def getStrategy(self):
        return(self.strategy)
    
    def introduce(self):
        if verbose:
            print("{0} agent {3} born with strategy p={1:.4f}, q={2:.4f}".format(self, self.strategy['offer'], self.strategy['accept'], self.id))
        
        if logging == True:
            if latexTable == True:
                lafile.write("\t \\multicolumn{{4}}{{l}}{{\\texttt{{{0} initialised $s_{{{1}}}=({2:.4f},{3:.4f})$ }} }}\\\ \n".format(self, self.id, self.strategy['offer'], self.strategy['accept']))
            file.write("{0} born with strategy p={1:.4f}, q={2:.4f}\n".format(self, self.strategy['offer'], self.strategy['accept']))
            
        
    def budgeting(self, payoff, role, partner):                               # revenue per round
        #self.revenue.append(payoff)
        self.revenue += payoff
        if verbose:
            if role == "proposer":
                print("{0}: payoff {1}, partner {2}".format(self, round(payoff, 2), partner))
        
    def storeMoney(self, currentRound):
        income = self.revenue #sum(self.revenue)
        #self.wallet.append(income)#self.wallet.append(round(np.sum(self.revenue), 2))
        self.stratIncome.append(income)#self.stratIncome.append(round(np.sum(self.revenue), 2))
        
        self.fitness = income / (2 * self.degree)#len(self.neighbours))#np.mean(self.stratIncome) / (2* len(self.neighbours)) 
        
        if dataStore == True or currentRound == rounds-1:
            self.data.append([self.strategy['offer'], self.strategy['accept'], income])

        #print("{0} revenue: {1}, sI: {2}, sI-mean: {3}, fit: {4}, K: {5}".format(self, self.revenue, self.stratIncome, np.mean(self.stratIncome), self.fitness, len(self.neighbours)))
        
        if self.fitness > 1.0:
            raise ValueError("fitness no bueno chef")
                
        
    def findExemplar(self):
        paydict = {}
        
        for neighbour in self.neighbours:
            paydict[neighbour] = neighbour.fitness
        
        best = max(paydict,key = paydict.get)
        #print("{0} fit: {1} pay: {4}, best {2} fit: {3} pay: {5}".format(self, self.fitness, best, best.fitness, self.revenue, best.revenue))
        #file.write("{0} fit: {1:.4f} pay: {4:.4f}, best {2} fit: {3:.4f} pay: {5:.4f}\n".format(self, self.fitness, best, best.fitness, self.revenue, best.revenue))
        
        if best.fitness > self.fitness:
            self.exemplar = best
        elif best.fitness == self.fitness:
            self.exemplar = choice([self, best])
        else:
            self.exemplar = self
        
        
    def updateStrategy(self, currentRound):
        #if random.random() < explore:
        #    self.nextstrat = self.randomise()
        #    print("round {3}: {0} exploring and taking new strategy p = {1:0.2f}, q = {2:0.2f}".format(self, self.strategy["offer"], self.strategy["accept"], currentRound))   
        #else:
        self.comparisonMethods(selectionStyle)(currentRound)
        
    
    def unconditional(self, currentRound):
        self.findExemplar()
        self.nextstrat = self.exemplar.strategy
        
        if self != self.exemplar:
            #if verbose:
            #print("round {4}: {0} exploiting strategy from {1}: p = {2:0.2f}, q = {3:0.2f}".format(self, self.exemplar, self.strategy["offer"], self.strategy["accept"], currentRound))
            if logging == True:
                if latexTable == True:
                    #lafile.write("\t& {0} & \\multicolumn{{2}}{{l}}{{ exploit {4} with $s_{{{7}}}: p = {5:0.4f}, q = {6:0.4f}$}} \\\ \n".format(self, self.id, self.strategy['offer'], self.strategy['accept'], self.exemplar, self.exemplar.strategy["offer"], self.exemplar.strategy["accept"], self.exemplar.id))
                    lafile.write("\t& {0} & exploit {1} & \\\ \n".format(self, self.exemplar))
                file.write("{0} exploiting strategy from {1}: p = {2:0.4f}, q = {3:0.4f}\n".format(self, self.exemplar, self.exemplar.strategy["offer"], self.exemplar.strategy["accept"], currentRound))
            
    
    def proportional(self, currentRound):
        model = choice(self.neighbours)
        # note! neighbour selection is now random instead of ((( the best )))
        
        revSelf = self.fitness#np.mean(sum(self.revenue))#self.stratIncome)
        revOpp = model.fitness#np.mean(sum(model.revenue))#model.stratIncome)
              
        changeProb = (revOpp - revSelf)# / (2 * max(self.degree, model.degree))#len(self.neighbours), len(model.neighbours)))
        
        if verbose:
            print("{0} revSelf: {1}, {2} revOpp: {3}, changeProb = {4}, k = {5}".format(self, revSelf, model, revOpp, changeProb, max(self.degree, model.degree)))#max(len(self.neighbours), len(model.neighbours))))
        if changeProb > 1.0:    
            raise ValueError("impossible changeprob")
        prob = random.random()
        
        if prob < changeProb:
            self.nextstrat = model.strategy
            if verbose:
                print("{0} switch!".format(self))
            if logging == True:
                if latexTable == True:
                    lafile.write("\t& {0} & exploit {1} & with $\\pi_{{{4}, {5}}} = {3:.4f}$\\\ \n".format(self, model, prob, changeProb, self.id, model.id))
                file.write("{0} exploiting strategy from {1}: p = {2:0.4f}, q = {3:0.4f}\n".format(self, self.exemplar, self.exemplar.strategy["offer"], self.exemplar.strategy["accept"], currentRound))
            
            
    def fermi(self, currentRound):
        model = choice(self.neighbours)
        fermiProb = 1 / (1 + np.exp(-selectionIntensity * (model.fitness - self.fitness)))
        prob = random.random()
        if prob < fermiProb:
            self.nextstrat = model.strategy
            if verbose:
                print("& {0} fitSelf: {1}, {2} fitMod: {3}, fermiProb: {4}".format(self, self.fitness, model, model.fitness, fermiProb))
                print("round {4}: {0} changing strategy ({1}) to that of {2}: {3}".format(self, self.strategy, model, model.strategy, currentRound))
            if logging == True:
                if latexTable == True:
                    lafile.write("\t& {0} & exploit {1} & with $\\pi_{{{4}, {5}}} = {3:.4f}$\\\ \n".format(self, model, prob, fermiProb, self.id, model.id))
                file.write("{0} exploiting strategy from {1}: p = {2:0.4f}, q = {3:0.4f}\n".format(self, self.exemplar, self.exemplar.strategy["offer"], self.exemplar.strategy["accept"], currentRound))

    def comparisonMethods(self, argument):
        switcher = {
            "unconditional": self.unconditional,
            "proportional": self.proportional,
            "Fermi": self.fermi
        }
        pairFunc = switcher.get(argument, lambda: "Invalid")
        return(pairFunc)
    
    
    def randomise(self):
        strategy = {}
        strategy["offer"] = random.uniform(0, 1)#round(random.uniform(0, 1), 2)#choice(list(range(1,10,1)))/10
        strategy["accept"] = random.uniform(0, 1)#round(random.uniform(0, 1), 2)#choice(list(range(1,10,1)))/10
        if strategy["offer"] > 1.0:
            raise ValueError("randomise screws up offer")
        if strategy["accept"] > 1.0:
            raise ValueError("randomise screws up accept")
        return(strategy)
    
    
    def exploration(self):
        if random.random() < explore:
            self.nextstrat = self.randomise()
            #print("{0} exploring and changing from {1} to {2}".format(self, self.strategy, self.nextstrat))
            if logging == True:
                if latexTable == True:
                    #file.write("\t& {0} & $s_{{{1}}}=({2},{3})$ & exploring: new $p={4:.4f}$, $q={5:.4f}$\\\ \n".format(self, self.id, self.strategy['offer'], self.strategy['accept'], self.nextstrat['offer'], self.nextstrat['accept']))
                    lafile.write("\t& {0} & exploring & new $p={1:.4f}$, $q={2:.4f}$\\\ \n".format(self, self.nextstrat['offer'], self.nextstrat['accept']))
                file.write("{0} exploring from p={1:.4f}, q={2:.4f} to p={3:.4f}, q={4:.4f}\n".format(self, self.strategy['offer'], self.strategy['accept'], self.nextstrat['offer'], self.nextstrat['accept']))
                    
            self.strategy = self.nextstrat
            
    """        
    def explorationtest(self):
        if random.random() < 1:#explore:
            strategy = {}
            strategy["offer"] = 0.3#random.uniform(0, 1)#round(random.uniform(0, 1), 2)#choice(list(range(1,10,1)))/10
            strategy["accept"] = 0.7#random.uniform(0, 1)
            self.nextstrat = strategy
            print("new strat {0}".format(self.nextstrat)) 
            #print("{0} exploring and changing from {1} to {2}".format(self, self.strategy, self.nextstrat))
            if logging == True:
                if latexTable == True:
                    #file.write("\t& {0} & $s_{{{1}}}=({2},{3})$ & exploring: new $p={4:.4f}$, $q={5:.4f}$\\\ \n".format(self, self.id, self.strategy['offer'], self.strategy['accept'], self.nextstrat['offer'], self.nextstrat['accept']))
                    lafile.write("\t& {0} & exploring & new $p={1:.4f}$, $q={2:.4f}$\\\ \n".format(self, self.nextstrat['offer'], self.nextstrat['accept']))
                file.write("{0} exploring from p={1:.4f}, q={2:.4f} to p={3:.4f}, q={4:.4f}\n".format(self, self.strategy['offer'], self.strategy['accept'], self.nextstrat['offer'], self.nextstrat['accept']))
                    
            self.strategy = self.nextstrat
    """        
            
    def noisify(self, oldstrategy):
        newstrategy = {}
        #ensure that p and q are in [0, 1]
        alpha = noise_e/2
        p_alpha = random.uniform(-alpha, alpha)
        q_alpha = random.uniform(-alpha, alpha)
        p = min(max(oldstrategy["offer"] + p_alpha, 0), 1)#random.uniform(-alpha, alpha), 0), 1.0)
        q = min(max(oldstrategy["accept"] + q_alpha, 0), 1)#random.uniform(-alpha, alpha), 0), 1.0)
        #file.write("p_alpha = {0:.5f}, q_alpha = {1:.5f}\n".format(p_alpha, q_alpha))
        newstrategy["offer"] = p
        newstrategy["accept"] = q
        
        return(newstrategy)
    
    
    def changeStrat(self):
        if self.strategy != self.nextstrat:
            self.stratIncome.clear()
            
            if noise:
                self.strategy = self.noisify(self.nextstrat)
                if verbose:
                    print("{0} noisified from {1} to {2}".format(self, self.nextstrat, self.strategy))
            else:   
                self.strategy = self.nextstrat
            
            self.nextstrat = self.strategy
            
        
    def shareData(self):
        return(self.data)
            
    
    def shareStats(self):
        stats = [self.strategy['offer'], self.strategy['accept'], self.revenue]#sum(self.revenue)] # share stats every round #or use mean?
        return(stats)
    
    
    def clear(self):
        #if len(self.revenue) % 2 != 0:
        #    raise ValueError("revenue no bueno chef")
        #self.revenue.clear()
        self.revenue = 0


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
         
         
    def createSFN(self):
        
        m_step = np.linspace(1, agentCount, simulations, endpoint=False)
        
        #if int(m_step[i]) < agentCount:
        #    m = int(m_step[i])
        #else:
        #    m = (agentCount-1)
        
        m = 3#m_step[12]
        
        # --> nx.extended_barabasi_albert_graph?
        SFN = nx.barabasi_albert_graph(agentCount, m) # scale-free network characterised by having vastly differing degrees (hence scale-free), small amount of large hubs
        
        SFNcharacteristics = Graph.graphCharacteristics(SFN)
        
        fnameSFN = "Barabasi-Albert_n{0}_sim{1}_m={2}".format(agentCount, simulations, m)
        
        #nx.write_edgelist(SFN, "Graphs/Barabasi-AlbertV{0}_n{1}_sim{2}_round{3}_exp={4:.2f}_prop={5}_random={6}".format(i, agentCount, simulations, rounds, explore, str(randomRoles)))
        
        nx.write_gpickle(SFN, "Graphs/{0}.gpickle".format(fnameSFN))
        
        
        #self.graphData[fnameSWN], self.graphData[fnameSFN] = SWNcharacteristics, SFNcharacteristics
        self.graphData[fnameSFN] = SFNcharacteristics
        
        if showGraph:
            fnameSFN = "Barabasi-AlbertV_n{0}_sim{1}_m={2}".format(agentCount, simulations, m)
            
            #nx.draw(graph, node_color='r', with_labels=True, alpha=0.53, width=1.5)
            nx.draw_kamada_kawai(SFN, with_labels=True, edge_color = '#00a39c', node_color = '#ff6960', alpha=0.63, node_size = 200, width = 1)
            plt.title('{0} m = {3:0.3f}, APL={4:0.3f}, CC = {5:0.3f}, SP = {6:0.3f})'
                      .format('Barabàsi-Albert', i, simulations, m, gg.graphData[fnameSFN]['APL'], gg.graphData[fnameSFN]['CC'], gg.graphData[fnameSFN]['SPavg']))

            plt.show()
        self.graphs.append(SFN)
            
        #print("graphs created")
    
        return(self.graphs, self.graphData)
            
        """
        if showGraph:
            for key in self.graphData.keys():
                if key == 'Watts-Strogatz':
                    x = p_step
                    xlab = 'probability of rewiring random edge'
                if key == 'Barabasi-Albert':
                    x = m_step
                    xlab = 'degree for each new agent'
                Plot().measurePlot(key, x, gg.graphData, xlab)
        """
        
    def createSWN(self, p):
                
        SWN = nx.connected_watts_strogatz_graph(agentCount, edgeDegree, p)
        
        SWNcharacteristics = Graph.graphCharacteristics(SWN)
        
        fnameSWN = "Watts-Strogatz_n{0}_k={2}_p={3}_APL={4}_CC={5}_SP={6}".format(agentCount, simulations, edgeDegree, p, SWNcharacteristics['APL'], SWNcharacteristics['CC'], SWNcharacteristics['SPavg'])
        
        #nx.write_edgelist(SWN, "Graphs/Watts-StrogatzV{0}_n{1}_sim{2}_round{3}_exp={4:.2f}_prop={5}_random={6}".format(i, agentCount, simulations, rounds, explore, str(randomRoles)))
        nx.write_gpickle(SWN, "Graphs/Experiment 1/{0}.gpickle".format(fnameSWN))        
        
        self.graphData[fnameSWN] = SWNcharacteristics
        
        if showGraph:
            #fnameSWN = "Watts-StrogatzV{0}_n{1}_sim{2}_k={3}_p={4}".format(i, agentCount, simulations, edgeDegree, p)
            
            nx.draw_kamada_kawai(SWN, with_labels=True, edge_color = '#00a39c', node_color = '#ff6960', alpha=0.63, node_size = 200, width = 1)
            plt.title('{0}, p = {3:0.3f}\nAPL={4:0.3f}, CC = {5:0.3f}, SP = {6:0.3f})'
                      .format('Watts-Strogatz', i, simulations, p, gg.graphData[fnameSWN]['APL'], gg.graphData[fnameSWN]['CC'], gg.graphData[fnameSWN]['SPavg']))
                              #p_step[i], gg.graphData[fnameSWN]['APL'], gg.graphData[fnameSWN]['CC'], gg.graphData[fnameSWN]['SPavg']))
            plt.show()
        
        #self.graphs.append(SWN)   
        #print("graphs created")
    
        return(SWN, self.graphData[fnameSWN]) #self.graphs, self.graphData[fnameSWN])

        
    def graphCharacteristics(g):
        APL = nx.average_shortest_path_length(g)    # average of all shortest paths between any node couple
        clust = nx.average_clustering(g)
        SPgraph, SPtotal = Graph.structuralPower(g)
        SPtotal = [round(SP, 4) for SP in SPtotal]
        
        charList = OrderedDict([('APL', APL), ('CC', clust), ('SPavg', SPgraph), ('SPnodes', SPtotal)])
        # for deg dist see https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_degree_histogram.html
        return charList

    def structuralPower(g):
        SPtotal = list(range(agentCount))
        groups = dict()
        
        for node in g.nodes:            
            groups[node] = set(g.neighbors(node))
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
        birth = 0
        while birth < agentCount:
            agent = Agent(self.graph)
            agent.introduce()
            self.agents.append(agent)
            birth += 1
        if logging == True:
            if latexTable == True:
                lafile.write("\t &&& \\\ \n")          
        for agent in self.agents:
            agent.meetNeighbours(self.graph)
            
    def returnAgents(self):
        if updating == 1:
            agentPoule = sample(self.agents, k=updateN)
            if verbose:
                print("Agent(s) for updating: {0}".format(agentPoule))
        else:
            agentPoule = self.agents
        return(agentPoule)
               
    def killAgents(self):
        Agent.agentID = 0   
        for agent in self.agents:
            agent.kill()

           
class ultimatumGame:
    
    def __init__(self, graph):
        self.population = Population(graph)
        self.graph = self.population.graph
        self.edges = self.graph.edges
        
        if edgeDegree >= agentCount:
            raise ValueError("Amount of edges per node cannot be equal to or greater than amount of agents")        
    
    def game(self, proposer, responder, currentRound):         # the actual interaction between prop and resp
        if testing:
            print("Ultimatum Game between (proposer) {0} and (responder) {1}:"
                  .format(proposer, responder))
            print("\tP {0} strategy {1} and R {2} strategy {3}".format(proposer, proposer.strategy, responder, responder.strategy))
        
        offer = proposer.getStrategy()['offer']
        accept = responder.getStrategy()['accept']

        if offer > 1.0:
            raise ValueError("{0} got offer strategy of 1.0 ({1})".format(proposer, offer))
        if accept > 1.0:
            raise ValueError("{0} got accept strategy of 1.0({1})".format(responder, offer))
        
        if offer >= accept:
            success = 1
            payPro = 1 - offer
            payRes = offer
        else:
            success = 0
            payPro = 0
            payRes = 0
            if testing:
                print("Offer {0} ({1}) too low for acceptance {2} ({3})"
                      .format(offer, proposer, accept, responder))
        
        #print("{0} p={1}, {2} q={3}, suc={4}".format(proposer, offer, responder, accept, success))
        
        self.edges[proposer.id, responder.id][currentRound].append([offer, accept, success])
        
        proposer.budgeting(payPro, "proposer", responder)
        responder.budgeting(payRes, "responder", proposer)

    
    
    def play(self):                              # selection of players and structure surrounding an interaction    
        self.population.populate()
        
        struct = self.graph
        nodes = struct.nodes         
        
        for n in range(rounds):
            if n+1 % 1000 == 0:
                print("  == round {0} ==".format(n))
            if logging == True:
                if latexTable == True:
                    lafile.write("\t{0}".format(n+1))
                file.write("\n === R O U N D {0} ===\nagent 0 = ({1:.4f}, {2:.4f}); agent 1 = ({3:.4f}, {4:.4f})\n".format(n, self.population.agents[0].strategy['offer'], self.population.agents[0].strategy['accept'], self.population.agents[1].strategy['offer'], self.population.agents[1].strategy['accept']))
            
            for edge in self.edges:
                players = edge
                
                self.edges[edge][n] = []
                
                for i in range(2):
                    #j = choice(range(2))
                    #proposer = nodes[proposers[j]]['agent']
                    #responder = nodes[responders[j]]['agent']
                    if randomRoles:
                        random.shuffle(players)
                    proposer = nodes[players[i]]['agent']
                    responder = nodes[players[i-1]]['agent']
                    self.game(proposer, responder, n)
                
                if logging == True:
                    if latexTable == True:
                        #file.write("\t& {0} & $s_{{{1}}}=({2:.4f},{3:.4f})$ & offer ${2:.4f}$ to {4} accept: ${5:.4f}$ success = {6} \\\ \n".format(proposer, proposer.id, proposer.strategy['offer'], proposer.strategy['accept'], responder, responder.strategy['accept'], success))
                        lafile.write("\t& Agent {0} & Agent {1} & success = {2}\\% \\\ \n".format(edge[0], edge[1], (mean((self.edges[edge][n][0][-1], self.edges[edge][n][1][-1]))*100)))
                
            for agent in self.population.agents:
                agent.storeMoney(n)
                if logging == True:
                    if latexTable == True:
                        lafile.write(" \t&{0} & $s_{{{1}}}=({2:.4f},{3:.4f})$ & $f_{{{1}}}={4:.4f}$, $u_{{{1}}}={5:.4f}$ \\\ \n".format(agent, agent.id, agent.strategy['offer'], agent.strategy['accept'], agent.fitness, agent.revenue))
                        
            if logging == True:
                if latexTable == False:
                    lafile.write("{0} fit: {1:.4f} pay: {4:.4f}, best {2} fit: {3:.4f} pay: {5:.4f}\n".format(self.population.agents[0], self.population.agents[0].fitness, self.population.agents[1], self.population.agents[1].fitness, self.population.agents[0].revenue, self.population.agents[1].revenue))
            
            if n != (rounds - 1):
                self.updateAgents(n)
                
                if latexTable == True:
                    lafile.write("\hline\n")
       
        # ==== end of rounds =====
        
        if dataStore == False:
            agentdata = [[agent.shareData()[-1]] for agent in self.population.agents]
            #agentdata = [[agentdata[ag][-1]] for ag in range(agentCount)]
            edgedata =  np.zeros((len(edgeList), 1, 2, 3), dtype='float16')
        else:
            agentdata = [agent.shareData() for agent in self.population.agents]
            edgedata = [list(struct.get_edge_data(*edge).values()) for edge in struct.edges]
        
        #print("see here {0}".format(np.shape(agentdata)))
        #print("values for p: {0} \nvalues for q: {1}".format(np.array(agentdata).T[0][-1], np.array(agentdata).T[1][-1]))
        
        
        return(agentdata, edgedata)
    
    
    def updateAgents(self, n):
        updagents = self.population.returnAgents()
        
        for agent in updagents:
            agent.updateStrategy(n)
            
        for agent in updagents:
            agent.changeStrat()
        
        for agent in self.population.agents:
            #if agent == self.population.agents[0] and n == 21:
            #    agent.explorationtest()
            #if n > 50:
            agent.exploration()
            agent.clear()


class Plot:

    def __init__(self):
        self.name = "None"

    def measurePlot(self, key, xval, graphCharacteristics, xlab):
        
        yAPL = list()
        yCC = list()
        
        for sim in range(simulations):
            yAPL.append(graphCharacteristics[key][sim]['APL'])
            yCC.append(graphCharacteristics[key][sim]['CC'])
        
        plt.plot(xval, yAPL, 'D', color = 'r', linewidth = 0.5, markersize = 3, label = "APL")
        plt.plot(xval, yCC, '2', color = 'b', linewidth = 0.5, markersize = 5, label = "CC")
        plt.title('Average Path Length and Clustering Coefficient for ' + key)
        plt.legend(loc='center right', shadow=True, ncol=1)
        plt.xlabel(xlab)
        plt.savefig('Graphs/Characteristics Plot {0}(sim{1})'.format(key, simulations))
        plt.show()
        
    def doPlot(self, x, y, err, name):
        plt.plot(x,y, '.-', color= 'black', linewidth = 0.5, markersize = 3)#, pointwidth = 0.3)
        plt.title(name)
        plt.fill_between(x, y + err, y - err, alpha=0.2, color = 'r')
        plt.ylim([0,1])
        plt.xlabel('rounds (n)')
        plt.show()
               
    
class Simulation:
    
    def __init__(self):
        self.finalPlot = Plot()
    
    def run(self):        
        if dataStore == False:
            gameTest = np.zeros((agentCount, 1, 3, simulations))
            edgeTest = np.zeros((len(edgeList), 1, 2, 3, simulations), dtype='float16')
        else:
            gameTest = np.zeros((agentCount, rounds, 3, simulations), dtype='float16')
            edgeTest = np.zeros((len(edgeList), rounds, 2, 3, simulations), dtype='float16')
        
        def playUG():
            print('\n=== simulation {0} ==='.format(sim))
            UG = ultimatumGame(graph)
            gameTest[:, :, :, sim], edgeTest[:, :, :, :, sim] = UG.play()
            UG.population.killAgents()
                                    
        for sim in range(simulations):
            playUG()
        
        print(np.shape(gameTest))
        print(np.shape(edgeTest))
        
        gameTest = gameTest.mean(axis=3)
        edgeTest = edgeTest.mean(axis=4)
        
        gameTest = np.transpose(gameTest, (1, 0, 2))
        edgeTest = np.transpose(edgeTest, (1, 0, 2, 3))
    
        return(gameTest, edgeTest)                
            
                


def dataHandling(gameTest, edgeTest):
    
    ### index: sims / agents / rounds / (entries) / variables   
    ### construct MultiIndex values to save as pd DataFrame
    
    #indexGame = pd.MultiIndex.from_product((graphType, range(simulations), agentList, ['p', 'q', 'u']), names=['Graph', 'Simulation', 'Agent', 'value']) #graphType, range(simulations), agentList), names=['Graph', 'Simulation', 'Agent'])
    #indexEdge = pd.MultiIndex.from_product((graphType, range(simulations), edgeList, ['0', '1'], ['p', 'q', 'suc']), names=['Graph', 'Simulation', 'Edge', 'interact', 'value'])
    indexGame = pd.MultiIndex.from_product((graphType, agentList, ['p', 'q', 'u']), names=['Graph', 'Agent', 'value'])
    indexEdge = pd.MultiIndex.from_product((graphType, edgeList, ['0', '1'], ['p', 'q', 'suc']), names=['Graph', 'Edge', 'interact', 'value'])
        
    gameTest = gameTest.reshape(rounds, -1)
    edgeTest = edgeTest.reshape(rounds, -1)
    
    gameData = pd.DataFrame(data=gameTest, index = range(rounds), columns = indexGame)                
    edgeData = pd.DataFrame(data=edgeTest, index = range(rounds), columns = indexEdge)
    
    gameData.columns = gameData.columns.map(str)
    edgeData.columns = edgeData.columns.map(str)
    
    gameData.to_parquet("Data/gameData_n{0}_sim{1}_round{2}_exp={3:.4f}_random={4}_select={5}_beta={6}_updating={7}_updateN={8}.parquet".format(agentCount, simulations, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN))
    edgeData.to_parquet("Data/edgeData_n{0}_sim{1}_round{2}_exp={3:.4f}_random={4}_select={5}_beta={6}_updating={7}_updateN={8}.parquet".format(agentCount, simulations, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN))
    

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
        
        gs2 = ax2.get_gridspec()
        ax2.remove()
        ax3.remove()
        axplot = fig.add_subplot(gs[0, 1:3])
        xval = np.arange(0, rounds, 1)        
        nx.draw_networkx(graph, pos = positions, ax=axgraph, edge_color = edgecol, edge_cmap = plt.cm.coolwarm, node_color = nodecol, edge_vmin=0, edge_vmax=(2*rounds), alpha = 0.53, node_size = 1200, width = edgesize, with_labels=True, font_size = 30)
        
        # used to be ax2
        axplot.set_ylim([0, 1])
        axplot.set_xlim([0, rounds-1])
        axplot.set_xticks(np.append(np.arange(1, rounds, step=math.floor(rounds/20)), rounds))
        fairline = axplot.axhline(0.5, color="black", ls='--', alpha=0.4)
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
    
def degreeplot(totaldat, degrees):
    totalp = []
    totalq = []
    totalAPL = []
    totalCC = []
    totalSP = []
    
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
    
    #plt.plot(totalAPL, 'g^',  totalCC, 'y*', totalSP, 'm.')
    

def betaplot(totaldat, betalist):
    totalp = []
    totalq = []
    totalAPL = []
    totalCC = []
    totalSP = []
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
    #totalCC = 
    #totalSP = 
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
    rounds = 10000
    agentCount = 60
    edgeDegree = 4
    
    selectionStyle = "Fermi"      # 0: unconditional, 1: proportional, 2: Fermi
    selectionIntensity = 10 # the bèta in the Fermi-equation
    
    explore = 0.005       # with prob [explore], agents adapt strategy randomly. prob [1 - explore] = unconditional/proportional imit

    # =============================================================================
    # GAME SETTINGS
    # =============================================================================
    
    randomRoles = False  # if false, focal agent is assigned role randomly
  
    noise = True        # noise implemented as [strategy to exploit] ± noise_e 
    noise_e = 0.01 
    
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
    
    #graph = nx.complete_graph(agentCount)
    
    totaldata = []
    times = []
    degrees = []
    
    #gg = Graph()
    #graphs, gdata = gg.createGraph()
    
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
    betaList = []
    
    for selectionIntensity in [10, 20, 30]:
        
        print("current run: beta = {0}".format(selectionIntensity))
        betaList.append(selectionIntensity)
        
        p = 0.1
        
        gg = Graph()
        graph, gdata = gg.createSWN(p)
        
        #graph = graphs[0]
        g = 'Watts-Strogatz'
        
        positions = nx.kamada_kawai_layout(graph)
        #read = open("Graphs/{0}V{1}_n{2}_sim{3}_k={4}_p={5}.gpickle".format(g, i, agentCount, simulations, k, p), 'rb')#p_step[i]), 'rb')
        #graph = nx.read_gpickle(read)
        
        edgeList = [str(edge) for edge in graph.edges]
        agentList = np.array(["Agent %d" % agent for agent in range(0,agentCount)])
        
        if len(graph) != agentCount:
           raise ValueError("agents incorrectly replaced")
        
        start = process_time()
        
# =============================================================================
#         if logging == True:
#             file=open("Logs/{0}V{1}_n{2}_round{3}_exp={4:.3f}_noise={10}_random={5}_select={6}_beta={7}_updating={8}_updateN={9}.txt".format(g, i, agentCount, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN, noise_e), 'w+')
#         if latexTable == True:
#             lafile=open("Latex/{0}V{1}_n{2}_round{3}_exp={4:.3f}_noise={10}_random={5}_select={6}_beta={7}_updating={8}_updateN={9}.txt".format(g, i, agentCount, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN, noise_e), 'w+')
# =============================================================================
      
        gamedat, edgedat = Simulation().run() 
        totaldata.append((gamedat[-1], edgedat[-1], *list(gdata.values())))
        #generateImg(graph, g, positions, gamedat, edgedat, i)
                
# =============================================================================
#         if logging == True:
#             file.close()
#         if latexTable == True:
#             lafile.close()
# =============================================================================
        
        stop = process_time()
        times.append(stop-start)
        
        if dataStore == True:
            dataHandling(gamedat, edgedat)
    
    betaplot(totaldata, betaList)
    #degreeplot(totaldata, degrees)
    
    print("average process time: {0} s\ntotal process time: {1} s\nall process times: {2}".format(mean(times), sum(times), times))
    
    
#%%
"""
simlist = [
    '_n20_sim100_round10000_exp=0.0050_random=False_select=unconditional_beta=10_updating=1_updateN=10_V=40.parquet',
    '_n20_sim100_round10000_exp=0.0050_random=False_select=proportional_beta=10_updating=1_updateN=10_V=40.parquet',
    '_n20_sim100_round10000_exp=0.0050_random=False_select=Fermi_beta=10_updating=1_updateN=10_V=40.parquet'
    ]
graph = nx.read_gpickle(open('Graphs/Watts-StrogatzV40_n20_sim100_k=5_p=0.36363636363636365.gpickle', 'rb'))
positions = nx.spring_layout(graph)

for rd in simlist:
    g = 'Watts-Strogatz'
    
    gameAna = pd.read_parquet('HPtest/gameData' + rd, engine='pyarrow')
    edgeAna = pd.read_parquet('HPtest/edgeData' + rd, engine='pyarrow')
    gameAna.columns = pd.MultiIndex.from_frame(pd.DataFrame([literal_eval(col) for col in gameAna.columns]))
    edgeAna.columns = pd.MultiIndex.from_frame(pd.DataFrame([literal_eval(col) for col in edgeAna.columns]))
    
    gamedata = np.array([np.array(rnd).reshape(agentCount,3) for rnd in np.array(gameAna[g])])
    edgedata = np.array([np.array(rnd).reshape(len(graph.edges), 2, 3) for rnd in np.array(edgeAna[g])])
    
    i = 40
    name = rd
    generateImg(graph, g, positions, gamedata, edgedata, i)
            
    
    
    totaldata.append((gamedata, edgedata))
"""
 #%%   
    

    #settings = [simulations, rounds, agentCount, edgeDegree, explore, randomRoles]

"""
agentCount = 100
gecko = nx.erdos_renyi_graph(100, 0.06)
bara = nx.barabasi_albert_graph(100, 5) # scale-free network characterised by having vastly differing degrees (hence scale-free), small amount of large hubs
watson = nx.connected_watts_strogatz_graph(100, 5, 0.8) # small-world network characterised by low 
len(gecko.edges)
while not nx.is_connected(bara):
    gecko = nx.erdos_renyi_graph(100, 0.06)

posi = nx.spring_layout(gecko)
nx.draw_kamada_kawai(watson, with_labels=False, edge_color = '#00a39c', node_color = '#ff6960', alpha=0.90, node_size = 200, width = 1)

# %%

bla = watson

degree_sequence = sorted([d for n, d in bla.degree()], reverse=True)  # degree sequence
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

fig, ax = plt.subplots()
plt.title("Watts-Strogatz Degree Histogram")
plt.ylabel("Count")
plt.xlabel("k")
ax.set_xticks([d + 0.4 for d in deg])
ax.set_xticklabels(deg)
plt.bar(deg, cnt, width=0.80, color='#00a39c', alpha=0.85)

APL = nx.average_shortest_path_length(bla)    # average of all shortest paths between any node couple
clust = nx.average_clustering(bla)
SPgraph, SPtotal = Graph.structuralPower(bla)
SPtotal = [round(SP, 4) for SP in SPtotal]
print(np.mean(SPtotal))

with open("Watts-Strogatzn5p080.txt", "w") as output:
    output.write("APL = {0}; CC = {1}; SPavg = {2}".format(APL, clust, np.mean(SPtotal)))
"""