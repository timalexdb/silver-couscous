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


class Agent:
    
    agentID = 0
    
    def __init__(self, graph):
        self.id = Agent.agentID
        self.name = "Agent " +str(self.id)
        Agent.agentID += 1
        
        self.node = graph[self.id]
        graph.nodes[self.id]['agent'] = self #agentID and nodeID correspond; agent accessible directly through node key
        #self.neighbours = set()
        
        self.strategy = Strategy().randomize()
        self.wallet = []        # wallet keeps track of total revenue
        self.revenue = []       # revenue keeps track of #gains per round
        self.successes = 0
        self.exemplar = 0
        self.data = []
        
        if str(self) in data2.columns:
            del data2[str(self)]
        data2[str(self)] = ""
    
    
    def __repr__(self):
        return("Agent %d" %self.id)# + "S({0},{1})".format(self.strategy['offer'], self.strategy['accept']))
        
    
    def getStrategy(self):
        return(self.strategy)
    
    
    
    def introduce(self):
        print("{0} born with strategy {1}".format(self, self.strategy))
            
        
    def budgeting(self, payoff, role, partner):                               # revenue per round
        self.revenue.append(payoff)
        if payoff > 0:
            self.successes += 1
            
        self.data.append((partner, payoff, role))        
        
        
    def adapt(self, graph):                                    # first check highest value, after update strategy
        
        self.wallet.append(round(np.sum(self.revenue), 2))
        payMax = [np.sum(self.revenue), self]                   # calc with sum or mean?
        
        neighbours = list(graph.neighbors(self.id))
        
        #while max(listofneighbours) == (previous max):
        #   maxNeighlist.append(max(listofneighbours))
        #   listofneighbours.remove(max(listofneighbours))
        
        
        if testing:
            print("\n{0}:".format(self))
        
        #maxN = map(lambda payM, neighR: neighR >= payM, payMax[0], 
        
        for n in neighbours:
            
            neighRevenue = [np.sum(graph.nodes[n]['agent'].revenue), graph.nodes[n]['agent']]

            if testing:
                print("{0} : revenue = {1}".format(graph.nodes[n]['agent'], round(neighRevenue[0], 2)))
                
            if payMax[0] < neighRevenue[0]:
                payMax = neighRevenue
            
            # if neighbours have similar revenue, none can be benefited by neighbour order. doesn't account for focal agent
            
            if payMax[0] is neighRevenue[0] and payMax[1] != self:
                choiceList = [payMax, neighRevenue]
                payMax = random.choice(choiceList)
                
                
                
        # exemplar is the neighbour that will be imitated
        self.exemplar = payMax[1]
        
        if testing:
            if payMax[0] > np.sum(self.revenue):
                print("exemplar of {0} is {1} with {2} over {3}".format(self, self.exemplar, payMax[0], np.sum(self.revenue))) #mean or sum?
                

    def updateStrategy(self, currentRound):
        self.revenue.clear()
        
        if testing:
            print("{0}: Revenue of best neighbour {1}: {2}".format(self, self.exemplar, np.mean(self.exemplar.wallet[currentRound])))
        
        #insert probability here
        revSelf = self.wallet[currentRound]
        revOpp = self.exemplar.wallet[currentRound]
        
        changeProb = (revOpp - revSelf) / max(len(self.node), len(self.exemplar.node))
        #print("this is changeprob: {0}, len({3}):{1}, len({4}):{2}".format(round(changeProb, 2), len(self.node), len(self.exemplar.node), self, self.exemplar))
        
        if random.random() < explore:
            self.strategy = Strategy().randomize()
            print("{0} exploring and taking new strategy {1}".format(self, self.strategy))
            
        elif proportional:
            if random.random() < changeProb:
                self.strategy = self.exemplar.strategy
        else:
            self.strategy = self.exemplar.strategy
            print("{0} exploiting strategy from {1}: {2}".format(self, self.exemplar, self.strategy))

          
    def shareStats(self):
        stats = [self.strategy['offer'], self.strategy['accept'], np.mean(self.revenue)] # share stats every round
        #print("this is self.data for {0}: {1}".format(self, self.data))
        return(stats)
    
    
    def kill(self):
        del self



class Graph:
     
    def __init__(self):
         self.agentCount = agentCount
         
         
    def createGraph(self):
        graph = nx.random_regular_graph(edgeDegree, agentCount, seed=None) # random graph characterised by mostly similar degree
        # scale-free network characterised by having vastly differing degrees (hence scale-free), small amount of large hubs
        # small-world network characterised by low 
        
        for line in nx.generate_edgelist(graph):
            print(line)
        print("graph created")
        print(graph.nodes)
        nx.write_edgelist(graph, "Graphs/test_graph")
        print("graph written")
        
        
        
    #def graphCharacteristics(self, graph):
        #connectivity = nx.all_pairs_node_connectivity
        
        #   assortativity (simil of conn in the graph w.r.t. degree or other attr)
        #   average neighbor degree (average size of neighbor-neighborhood for each neighbor j of agent i)
        #   
    # implement structural measures here! also store graph struct + information here so that graphClass can be run by itself and OFFER graphs(...)
    # (...)to program instead of necessarily being called by program.
    
    
    
    
       
class Population:
    
    def __init__(self, graph):
        
        self.agents = set()
        self.graph = graph
        
        """
        if demo is True:
            self.graph = nx.Graph()
        else:
            self.graph = nx.random_regular_graph(edgeDegree, agentCount, seed=None)
        """
        
    def populate(self):
        birth = 0
        while birth < agentCount:
            agent = Agent(self.graph)
            agent.introduce()
            self.agents.add(agent)
            birth += 1
            
            
    def killAgents(self):
        Agent.agentID = 0
        for agent in self.agents:
            agent.kill()
            
                        
        #for agent in self.agents:
        #    print("this is the amount of neighbours for agent {0}: {1}".format(agent.id, len(agent.node)))
        nx.draw(self.graph, node_color='r', with_labels=True, alpha=0.53, width=1.5)
        plt.show()


           
class ultimatumGame:
    
    
    def __init__(self, graph):
        self.population = Population(graph)
        
        self.data = np.zeros((rounds, 6), dtype=float)
        #self.data2 = pd.DataFrame(index=range(rounds))
        self.offerList = []
        self.acceptList = []
        self.payList = []
        
        self.graph = self.population.graph
        
        self.plotting = Plot(self.data)

        #data2 = str(self.population.agents)
        
        
        if edgeDegree >= agentCount:
            raise ValueError("Amount of edges per node cannot be equal to or greater than amount of agents")
    
    
    def getData(self):
        return(self.data)        
    
        
    def game(self, proposer, responder, currentRound):         # the actual interaction between prop and resp
        if testing:
            print("Ultimatum Game between (proposer) {0} and (responder) {1}:"
                  .format(proposer, responder))
            print("P {0} strategy {1} and R {2} strategy {3}".format(proposer, proposer.strategy, responder, responder.strategy))
            
        offer = proposer.getStrategy()['offer']
        accept = responder.getStrategy()['accept']
        
        if offer == 1.0:
            raise ValueError("{0} got offer strategy of 1.0 ({1})".format(proposer, offer))
        if accept == 1.0:
            raise ValueError("{0} got accept strategy of 1.0({1})".format(responder, offer))
        
        if offer >= accept:
            success = 1
            payPro = round(1 - offer, 1)
            payRes = offer   
            if testing:
                print("Payoffs for proposer {0} = {1} and for responder {2} = {3}"
                      .format(proposer, payPro, responder, payRes))
        
        else:
            success = 0
            payPro = 0
            payRes = 0
            if testing:
                print("Offer {0} ({1}) too low for acceptance {2} ({3})"
                      .format(offer, proposer, accept, responder))
        
        self.graph.edges[proposer.id, responder.id]['round {0}'.format(currentRound)] = [offer, accept, success]
        print("edgedata (offer, accept, success): " +str(self.graph.edges[proposer.id, responder.id]['round {0}'.format(currentRound)]))
        
        proposer.budgeting(payPro, "proposer", responder)
        responder.budgeting(payRes, "responder", proposer)
                
    
    def play(self):                              # selection of players and structure surrounding an interaction
        
        self.population.populate()
        
        struct = self.population.graph
        
        for n in range(rounds):
            print("\n=== round {0} ===".format(n+1))
            
            for edge in struct.edges:
                proposers = random.sample(edge, 2)
                responders = proposers[::-1]

                if randomPlay:
                    
                    proposer = struct.nodes[proposers[0]]['agent']
                    responder = struct.nodes[responders[0]]['agent']
                    # ensures that agents play similar amount of games as with non-random play
                    for i in range(2):
                        j = random.choice(range(2))
                        proposer = struct.nodes[proposers[j]]['agent']
                        responder = struct.nodes[responders[j]]['agent']
                        self.game(proposer, responder, n)
                        print("proposer {0} and responder {1}".format(proposer, responder))
                else:
                    
                    for i in range(len(proposers)):
                        proposer = struct.nodes[proposers[i]]['agent']
                        responder = struct.nodes[responders[i]]['agent']
                        self.game(proposer, responder, n)
                   

            for agent in self.population.agents:
                agent.adapt(struct)
            print("\n")
            
            for agent in self.population.agents:
                stats = agent.shareStats()
                
                data2.iloc[n, agent.id] = stats
                
                #idea: row = round, col = agent, input = all stats from each agent in that round
                
                self.offerList.append(stats[0])
                self.acceptList.append(stats[1])
                self.payList.append(stats[2])
                
                if n != (rounds - 1):
                    agent.updateStrategy(n)
                
            self.data[n, 0] = np.mean(self.offerList)
            self.data[n, 1] = np.var(self.offerList)
            self.data[n, 2] = np.mean(self.acceptList)
            self.data[n, 3] = np.var(self.acceptList)
            self.data[n, 4] = np.mean(self.payList)
            self.data[n, 5] = np.var(self.payList)
            
            
            self.offerList.clear()
            self.acceptList.clear()
            self.payList.clear()
        
        if testing:           
            for agent in self.population.agents:
                print("\ntotal wallet for {0}: {1}".format(agent, agent.wallet))
                print("final payoff for {0} is: {1}, average: {2}".format(agent, round(np.sum(agent.wallet),4), round((np.sum(agent.wallet)/agent.successes), 4)))
        
        self.plotting.offerPlot()
        self.plotting.acceptPlot()
        self.plotting.payPlot()
        
    
class Plot:
    
    def __init__(self, stats):
        self.name = "None"
        self.stats = stats
        
            
    def offerPlot(self):
        
        xval = range(rounds)
        yval = self.stats[0:rounds,0]
        err = self.stats[0:rounds,1]
        
        self.doPlot(xval, yval, err, "Average offer per round")
        
    def acceptPlot(self):
        xval = range(rounds)
        yval = self.stats[0:rounds,2]
        err = self.stats[0:rounds,3]
        
        self.doPlot(xval, yval, err, "Average accept per round")
        
    def payPlot(self):
        xval = range(rounds)
        yval = self.stats[0:rounds,4]
        err = self.stats[0:rounds,5]
        
        self.doPlot(xval, yval, err, "Average payoff per round")
        
    def doPlot(self, x, y, err, name):
        plt.plot(x,y, '.-', color= 'black', linewidth = 0.5, markersize = 3)#, pointwidth = 0.3)
        plt.title(name)
        plt.fill_between(x, y + err, y - err, alpha=0.2, color = 'r')
        plt.ylim([0,1])
        plt.xlabel('rounds (n)')
        plt.show()
                  
        
        
class Strategy:
    
    def __init__(self):
        self.name = "None"
        
    def randomize(self):
        strategy = {}
        strategy["offer"] = random.choice(list(range(1,10,1)))/10
        strategy["accept"] = random.choice(list(range(1,10,1)))/10
        if strategy["offer"] > 0.9:
            raise ValueError("randomize fucks up offer")
        if strategy["accept"] > 0.9:
            raise ValueError("randomize fucks up accept")
        return(strategy)
    

    
    
class Simulation:
    
    def __init__(self):
        
        self.data = np.zeros((rounds, 6, simulations), dtype=float) # n of rounds, n of agents, n of values, amount of sims
        self.finalPlot = Plot(self.data)
        
    
    def run(self):
        
        for sim in range(simulations):
            print("\n=== Commencing Simulation {0} ===\n".format(sim+1))
            
            read = open("Graphs/test_graph", 'rb')
            graph = nx.read_edgelist(read, nodetype=int)
            print(graph.nodes)
            
            UG = ultimatumGame(graph)
            UG.play()
            
            for line in nx.generate_edgelist(graph):
                print(line)
            
            UG.population.killAgents()
            
            #data3[sim] = data2
            #self.data[:,:,sim] = UG.getData()



simulations = 2
rounds = 5
agentCount = 2
edgeDegree = 1

# STOCHASTICITY
explore = 0.4       # with prob [explore], agents adapt strategy randomly. prob [1 - explore] = unconditional/proportional imit

proportional = True
randomPlay = True

testing = False
demo = False

agentList = np.array(["Agent %d" % agent for agent in range(0,agentCount)])

#datadx = pd.MultiIndex.from_product([np.arange(0, simulations), np.arange(0, rounds), agentList])


#dataMI = pd.DataFrame(np.zeros((40)),index = datadx)#np.random.randn(len(datadx), 1), index = datadx)
data2 = pd.DataFrame(index=range(rounds), columns = agentList)       # just been messing around with this. REMEMBER .ILOC()
data3 = dict()                                  # stores dataframes per simulation
#pd.DataFrame(index=range(simulations)) #can't get Data2 into Data3.

Graph().createGraph()

game = Simulation()
game.run()

#print(dataMI.head())
#print(dataMI[0,3])
# ideas: graph generator for e.g. difference in SP, CC, various graphs (random, regular well-mixed etc)

#test
