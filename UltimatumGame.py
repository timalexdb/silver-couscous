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
from matplotlib import cm
from matplotlib import colors
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.animation as ani


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
        self.nextstrat = self.strategy
        self.wallet = []        # wallet keeps track of total revenue
        self.revenue = []       # revenue keeps track of #gains per round
        self.stratIncome = []
        self.fitness = 0
        self.successes = []
        self.exemplar = self
        self.data = []
    
    
    def __repr__(self):
        return("Agent %d" %self.id)# + "S({0},{1})".format(self.strategy['offer'], self.strategy['accept']))
    
    def meetNeighbours(self, graph):
        #print("{0} meeting neighbours! : {1}".format(self, self.neighbours))
        self.neighbours = list(graph.nodes[n]['agent'] for n in graph.neighbors(self.id))
        #print("{0} met neighbours! : {1}".format(self, self.neighbours))
    
    def getStrategy(self):
        return(self.strategy)
    
    def introduce(self):
        if verbose:
            print("{0} born with strategy {1}".format(self, self.strategy))
            
        
    def budgeting(self, payoff, role, partner):                               # revenue per round
        self.revenue.append(payoff)
        if role == "proposer":
            if verbose:
                print("{0}: payoff {1}, partner {2}".format(self, round(payoff, 2), partner))
        if payoff > 0:
            self.successes.append(1)
        
        
    def storeMoney(self):        
        self.wallet.append(np.sum(self.revenue))#self.wallet.append(round(np.sum(self.revenue), 2))
        self.stratIncome.append(np.sum(self.revenue))#self.stratIncome.append(round(np.sum(self.revenue), 2))
        self.fitness = np.mean(self.stratIncome) / (2* len(self.neighbours)) #np.mean(self.stratIncome) / len(self.neighbours)
        self.data.append([self.strategy['offer'], self.strategy['accept'], np.sum(self.revenue)])

        #print("{0} revenue: {1}, sI: {2}, sI-mean: {3}, fit: {4}, K: {5}".format(self, self.revenue, self.stratIncome, np.mean(self.stratIncome), self.fitness, len(self.neighbours)))
        
        if self.fitness > 1.0:
            #print()
            raise ValueError("fitness no bueno chef")
                
        
    def findExemplar(self):
        paydict = {}
        
        for neighbour in self.neighbours:
            paydict[neighbour] = neighbour.fitness
        
        best = max(paydict,key = paydict.get)
        
        if best.fitness > self.fitness:
            self.exemplar = best
        elif best.fitness == self.fitness:
            self.exemplar = np.random.choice([self, best])
        
        
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
            if verbose:
                print("round {4}: {0} exploiting strategy from {1}: p = {2:0.2f}, q = {3:0.2f}".format(self, self.exemplar, self.strategy["offer"], self.strategy["accept"], currentRound))
    
    
    def proportional(self, currentRound):
        model = random.choice(self.neighbours)
        # note! neighbour selection is now random instead of ((( the best )))
        
        revSelf = np.mean(self.stratIncome)#self.wallet[currentRound]  #self.wallet[currentRound]/len(self.neighbours)
        revOpp = np.mean(model.stratIncome)#model.wallet[currentRound]  #self.exemplar.wallet[currentRound]/len(self.exemplar.neighbours)
              
        changeProb = (revOpp - revSelf) / (2 * max(len(self.neighbours), len(model.neighbours)))
        if verbose:
            print("{0} revSelf: {1}, {2} revOpp: {3}, changeProb = {4}, k = {5}".format(self, revSelf, model, revOpp, changeProb, max(len(self.neighbours), len(model.neighbours))))
        if changeProb > 1.0:    
            raise ValueError("impossible changeprob")
        
        if random.random() < changeProb:
            self.nextstrat = model.strategy
            if verbose:
                print("{0} switch!".format(self))
            
            
    def fermi(self, currentRound):
        model = random.choice(self.neighbours)
        fermiProb = 1 / (1 + np.exp(-selectionIntensity * (model.fitness - self.fitness)))

        if random.random() < fermiProb:
            self.nextstrat = model.strategy
            if verbose:
                print("{0} fitSelf: {1}, {2} fitMod: {3}, fermiProb: {4}".format(self, self.fitness, model, model.fitness, fermiProb))
                print("round {4}: {0} changing strategy ({1}) to that of {2}: {3}".format(self, self.strategy, model, model.strategy, currentRound))


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
        strategy["offer"] = random.uniform(0, 1)#round(random.uniform(0, 1), 2)#random.choice(list(range(1,10,1)))/10
        strategy["accept"] = random.uniform(0, 1)#round(random.uniform(0, 1), 2)#random.choice(list(range(1,10,1)))/10
        if strategy["offer"] > 1.0:
            raise ValueError("randomise screws up offer")
        if strategy["accept"] > 1.0:
            raise ValueError("randomise screws up accept")
        return(strategy)
    
    
    def exploration(self):
        if random.random() < explore:
            self.nextstrat = self.randomise()
            
    
    def noisify(self, oldstrategy):
        newstrategy = {}
        #ensure that p and q are in [0, 1]
        p = min(max(oldstrategy["offer"] + np.random.uniform(-noise_e, noise_e), 0), 1.0)
        q = min(max(oldstrategy["accept"] + np.random.uniform(-noise_e, noise_e), 0), 1.0)

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
    
    #def storeData(self):
        
    
    def shareStats(self):
        stats = [self.strategy['offer'], self.strategy['accept'], np.sum(self.revenue)] # share stats every round #or use mean?
        return(stats)
    
    
    def clear(self):
        if len(self.revenue) % 2 != 0:
            raise ValueError("revenue no bueno chef")
        self.revenue.clear()


    def kill(self):
        del self



class Graph:
     
    def __init__(self):
         self.agentCount = agentCount
         self.graphData = dict()
         self.char1 = []
         self.char2 = []
         self.charT = []
         
    def createGraph(self):
        
        m_step = np.linspace(1, agentCount, simulations, endpoint=False)
        p_step = np.linspace(0, 0.9, simulations)
        
        for v, i in enumerate(np.arange(0, simulations, step = 10)): #[0, 0.001, 0.01, 0.1, 1]):#
            
            p = p_step[i]
            
            #if int(m_step[i]) < agentCount:
            #    m = int(m_step[i])
            #else:
            #    m = (agentCount-1)
            m = 3#m_step[12]
            
            # --> nx.extended_barabasi_albert_graph?
            SFN = nx.barabasi_albert_graph(agentCount, m) # scale-free network characterised by having vastly differing degrees (hence scale-free), small amount of large hubs
            
            if i == 0:
                k = 2
            else:
                k = edgeDegree
            
            SWN = nx.connected_watts_strogatz_graph(agentCount, k, p) # small-world network characterised by low 
            
            SWNcharacteristics = Graph.graphCharacteristics(SWN)
            SFNcharacteristics = Graph.graphCharacteristics(SFN)
            
            #self.char1.append(SWNcharacteristics)
            #self.char2.append(SFNcharacteristics)
            
            fnameSWN = "Watts-StrogatzV{0}_n{1}_sim{2}_k={3}_p={4}".format(i, agentCount, simulations, k, p) # edgeDegree, p)
            fnameSFN = "Barabasi-AlbertV{0}_n{1}_sim{2}_k={3}".format(i, agentCount, simulations, edgeDegree)
            
            #nx.write_edgelist(SWN, "Graphs/Watts-StrogatzV{0}_n{1}_sim{2}_round{3}_exp={4:.2f}_prop={5}_random={6}".format(i, agentCount, simulations, rounds, explore, str(randomRoles)))
            #nx.write_edgelist(SFN, "Graphs/Barabasi-AlbertV{0}_n{1}_sim{2}_round{3}_exp={4:.2f}_prop={5}_random={6}".format(i, agentCount, simulations, rounds, explore, str(randomRoles)))
            
            nx.write_gpickle(SWN, "Graphs/{0}.gpickle".format(fnameSWN))#"round{3}_exp={4:.2f}_random={5}.gpickle".format(i, agentCount, simulations, rounds, explore, str(randomRoles)))
            nx.write_gpickle(SFN, "Graphs/{0}.gpickle".format(fnameSFN))#round{3}_exp={4:.2f}_random={5}.gpickle".format(i, agentCount, simulations, rounds, explore, str(randomRoles)))
            
            #if showGraph:
            #    nx.draw(SWN, node_color='r', with_labels=True, alpha=0.53, width=1.5)
            #    plt.title('{0} (simulation {1}/{2}, p = {3:0.2f}, APL={4:0.2f}, CC = {5:0.2f}, SP = {6:0.2f})'.format('Watts-Strogatz', i, simulations, p, SWNcharacteristics['APL'], SWNcharacteristics['CC'], SWNcharacteristics['SPavg']))
            #    plt.show()

            #nx.draw(graph, node_color='b', with_labels=True, alpha=0.53, width=1.5)
            #plt.show()
            #nx.draw(graph2, node_color='r', with_labels=True, alpha=0.53, width=1.5)
            #plt.show()
            
            self.graphData[fnameSWN], self.graphData[fnameSFN] = SWNcharacteristics, SFNcharacteristics #self.char1, self.char2
            print("data for {0}: \n{1}".format(fnameSWN, self.graphData[str(fnameSWN)]))
            
            
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
    print("graphs created")
        
        
    def graphCharacteristics(g):
        APL = nx.average_shortest_path_length(g)    # average of all shortest paths between any node couple
        clust = nx.average_clustering(g)
        SPgraph, SPtotal = Graph.structuralPower(g)
        SPtotal = [round(SP, 4) for SP in SPtotal]
        
        # but which nodes??? adjust!
        charList = {'APL' : APL, 'CC' : clust, 'SPavg' : SPgraph, 'SPnodes' : SPtotal}
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
            
        for agent in self.agents:
            agent.meetNeighbours(self.graph)
            
    def returnAgents(self):
        if updating == 1:
            agentPoule = np.random.choice(self.agents, size=updateN, replace=False)
            if verbose:
                print("Agent(s) for updating: {0}".format(agentPoule))
        else:
            agentPoule = self.agents
            
        return(agentPoule)
        
            
    def killAgents(self):
        Agent.agentID = 0
        
        for agent in self.agents:
            agent.kill()


        #for agent in self.agents:
        #    print("this is the amount of neighbours for agent {0}: {1}".format(agent.id, len(agent.node)))
        #nx.draw(self.graph, node_color='r', with_labels=True, alpha=0.53, width=1.5)
        #plt.show()


           
class ultimatumGame:
    
    
    def __init__(self, graph):
        self.population = Population(graph)
        self.data = {}
        self.graph = self.population.graph           
        if edgeDegree >= agentCount:
            raise ValueError("Amount of edges per node cannot be equal to or greater than amount of agents")        
    
    
    def game(self, proposer, responder, currentRound):         # the actual interaction between prop and resp
        if testing:
            print("Ultimatum Game between (proposer) {0} and (responder) {1}:"
                  .format(proposer, responder))
            print("\tP {0} strategy {1} and R {2} strategy {3}".format(proposer, proposer.strategy, responder, responder.strategy))
        
        offer = proposer.getStrategy()['offer']
        accept = responder.getStrategy()['accept']
        
        if offer > 1.0: #== 1.0:
            raise ValueError("{0} got offer strategy of 1.0 ({1})".format(proposer, offer))
        if accept >1.0: #== 1.0:
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
        
        self.graph.edges[proposer.id, responder.id]['round {0}'.format(currentRound)].append([offer, accept, success])
        
        proposer.budgeting(payPro, "proposer", responder)
        responder.budgeting(payRes, "responder", proposer)

    
    
    def play(self, sim):                              # selection of players and structure surrounding an interaction    
        self.population.populate()
        
        struct = self.population.graph
        nodes = struct.nodes         
    
        for n in range(rounds):
            if n % 500 == 0:
                print("  == round {0} ==".format(n+1))
            
            for edge in struct.edges:
                proposers = random.sample(edge, 2)
                responders = proposers[::-1]
                
                self.graph.edges[nodes[edge[0]]['agent'].id, nodes[edge[1]]['agent'].id]['round {0}'.format(n)] = []
                #self.openEdge(struct.nodes[edge[0]]['agent'], struct.nodes[edge[1]]['agent'], n)
                
                if randomRoles:
                    # ensures that agents play similar amount of games as with non-random play                    
                    for i in range(2):
                        j = random.choice(range(2))
                        proposer = nodes[proposers[j]]['agent']
                        responder = nodes[responders[j]]['agent']
                        self.game(proposer, responder, n)
                else:
                    for i in range(len(proposers)):
                        proposer = nodes[proposers[i]]['agent']
                        responder = nodes[responders[i]]['agent']
                        self.game(proposer, responder, n)
                        
            for agent in self.population.agents:
                agent.storeMoney()
            
            if n != (rounds - 1):
                self.updateAgents(n)
       
        # ==== end of rounds =====
        
        agentdata = [agent.shareData() for agent in self.population.agents]
        edgedata = [list(struct.get_edge_data(*edge).values()) for edge in struct.edges]
        
        return(agentdata, edgedata)
    
    
    def updateAgents(self, n):
        updagents = self.population.returnAgents()
        
        for agent in updagents:
            agent.updateStrategy(n)
            
        for agent in updagents: #self.population.agents:
            agent.changeStrat()
        
        for agent in self.population.agents:
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
        self.data = np.zeros((rounds, agentCount, simulations), dtype=float) # n of rounds, n of agents, n of values, amount of sims
        self.finalPlot = Plot()
        
    
    def run(self):
        edgeList = []
        
        for g in graphType:
            # for type of network...
            
                p_step = np.linspace(0, 0.9, simulations)
            
            # v, i in enumerate([0.01, 1]):#[0, 0.001, 0.01, 0.1, 1]):#np.arange(0, simulations, step = 10):                
                # for network version the game is played on... 
                
                i = 40
                v = 40
                p = p_step[v]
                #read = open("Graphs/{0}V{1}_n{2}_sim{3}_round{4}_exp={5:.2f}_random={6}.gpickle".format(g, sim, agentCount, simulations, rounds, explore, str(randomRoles)), 'rb')                
                #read = open("Graphs/{0}V{1}_n{2}_sim{3}_k={4}_p={5}.gpickle".format(g, sim, agentCount, simulations, edgeDegree, p_step[i]), 'rb')
                
                if i == 0:
                    k = 2
                else:
                    k = edgeDegree
                
                read = open("Graphs/{0}V{1}_n{2}_sim{3}_k={4}_p={5}.gpickle".format(g, i, agentCount, simulations, k, p), 'rb')#p_step[i]), 'rb')
                graph = nx.read_gpickle(read)
                positions = nx.spring_layout(graph)
                
                edgeList = [str(edge) for edge in graph.edges]
                
                #if i > 0:
                #    sys.exit('done')
                
                if len(graph) != agentCount:
                    raise ValueError("agents incorrectly replaced")
                    
                if showGraph:
                    fnameSWN = "Watts-StrogatzV{0}_n{1}_sim{2}_k={3}_p={4}".format(i, agentCount, simulations, k, p)
                    
                    #nx.draw(graph, node_color='r', with_labels=True, alpha=0.53, width=1.5)
                    nx.draw_kamada_kawai(graph, with_labels=True, edge_color = '#00a39c', node_color = '#ff6960', alpha=0.63, node_size = 200, width = 1)
                    plt.title('{0} (graph v.{1}/{2}, p = {3:0.3f}, APL={4:0.3f}, CC = {5:0.3f}, SP = {6:0.3f})'
                              .format('Watts-Strogatz', i, simulations, p, gg.graphData[fnameSWN]['APL'], gg.graphData[fnameSWN]['CC'], gg.graphData[fnameSWN]['SPavg']))
                                      #p_step[i], gg.graphData[fnameSWN]['APL'], gg.graphData[fnameSWN]['CC'], gg.graphData[fnameSWN]['SPavg']))
                    plt.show()  #note that v*2 is pragmatical since p_step is created with a step of 10 instead of 20!
                
                if verbose:
                        print("characteristics of {0}({1}): \n{2}".format(g, i, gg.graphData[fnameSWN]))
                
                gameTest = np.zeros((agentCount, rounds, 3, simulations))
                edgeTest = np.zeros((len(edgeList), rounds, 2, 3, simulations))
                
                def playUG(sim):
                    print('\n=== simulation {0} ==='.format(sim))
                    UG = ultimatumGame(graph)
                    gameTest[:, :, :, sim], edgeTest[:, :, :, :, sim] = UG.play(sim)
                    UG.population.killAgents()
                                    
                for sim in range(simulations):
                    playUG(sim)
                                
                # index: sims / agents / rounds / (entries) / variables   
                ### construct MultiIndex values to save as pd DataFrame
                
                #indexGame = pd.MultiIndex.from_product((graphType, range(simulations), agentList, ['p', 'q', 'u']), names=['Graph', 'Simulation', 'Agent', 'value']) #graphType, range(simulations), agentList), names=['Graph', 'Simulation', 'Agent'])
                #indexEdge = pd.MultiIndex.from_product((graphType, range(simulations), edgeList, ['0', '1'], ['p', 'q', 'suc']), names=['Graph', 'Simulation', 'Edge', 'interact', 'value'])
                indexGame = pd.MultiIndex.from_product((graphType, agentList, ['p', 'q', 'u']), names=['Graph', 'Agent', 'value']) #graphType, range(simulations), agentList), names=['Graph', 'Simulation', 'Agent'])
                indexEdge = pd.MultiIndex.from_product((graphType, edgeList, ['0', '1'], ['p', 'q', 'suc']), names=['Graph', 'Edge', 'interact', 'value'])
                
                gameTest = gameTest.mean(axis=3)
                edgeTest = edgeTest.mean(axis=4)
                
                gameTest = np.transpose(gameTest, (1, 0, 2))
                edgeTest = np.transpose(edgeTest, (1, 0, 2, 3))
                
                self.generateImg(graph, g, positions, gameTest, edgeTest, i)
                
                gameTest = gameTest.reshape(rounds, -1)
                edgeTest = edgeTest.reshape(rounds, -1)
                
                gameData = pd.DataFrame(data=gameTest, index = range(rounds), columns = indexGame)                
                edgeData = pd.DataFrame(data=edgeTest, index = range(rounds), columns = indexEdge)
                
                gameData.columns = gameData.columns.map(str)
                edgeData.columns = edgeData.columns.map(str)
                
                gameData.to_parquet("Data/gameData_n{0}_sim{1}_round{2}_exp={3:.4f}_random={4}_select={5}_beta={6}_updating={7}_updateN={8}_V={9}.parquet".format(agentCount, simulations, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN, i))
                edgeData.to_parquet("Data/edgeData_n{0}_sim{1}_round{2}_exp={3:.4f}_random={4}_select={5}_beta={6}_updating={7}_updateN={8}_V={9}.parquet".format(agentCount, simulations, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN, i))


    def generateImg(self, graph, g, positions, gamedata, edgedata, i):
        
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
            varOffer = p_list.std(axis=0)
            avgAccept = q_list.mean(axis=0)
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
            edgecol = []#np.zeros(shape=(len(graph.edges), rounds))
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
            
            plt.savefig("Images/{0}V{1}_n{2}_round{3}_exp={4:.3f}_noise={10}_random={5}_select={6}_beta={7}_updating={8}_updateN={9}.png".format(g, i, agentCount, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN, noise_e))
            plt.close(fig)
        
        save_image()
       
    # %%
    # either use gameAna['graphtype']['sim']['agent(s)'][row:row+1] or gameAna[loc], see https://stackoverflow.com/questions/53927460/select-rows-in-pandas-multiindex-dataframe
    
    #   to change IPython backend, enter %matplotlib followed by 'qt' for animations or 'inline' for plot pane
    
    # agents increase in colour as their overall wallet sum increases relative to others

                
                
                
  
if __name__ == '__main__':
    # =============================================================================
    # HYPERPARAM
    # =============================================================================
    
    simulations = 100
    rounds = 10000
    agentCount = 20
    edgeDegree = 5
    
    selectionStyle = "Fermi"      # 0: unconditional, 1: proportional, 2: Fermi
    selectionIntensity = 10 # the bèta in the Fermi-equation
    
    explore = 0.005       # with prob [explore], agents adapt strategy randomly. prob [1 - explore] = unconditional/proportional imit

    # =============================================================================
    # GAME SETTINGS
    # =============================================================================
    
    randomRoles = False     # if false, focal agent is assigned role randomly
  
    noise = True        # noise implemented as [strategy to exploit] ± noise_e 
    noise_e = 0.005 #actually 0.01 since this is "r", not diam
    
    updating = 1            # 0 : all agents update; 1 : at random (n) agents update
    updateN = 10
    
    testing = False
    showGraph = True
    
    verbose = False

    gg = Graph()
    gg.createGraph()
    
    for selectionStyle in ['unconditional', 'proportional', 'Fermi']:
        
        #for explore in [0.001]:#, 0.005, 0.01]:#np.arange(0.001, 0.010, step = 0.5):#0.01, 0.08, step=0.02):
            
            #for updateN in [1, 5, 10]:
                
                agentList = np.array(["Agent %d" % agent for agent in range(0,agentCount)])
        
                graphType = ['Watts-Strogatz']#, 'Barabasi-Albert']
        
                print("{0} {1} {2}".format(selectionStyle, explore, updateN))
        
                Simulation().run()

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