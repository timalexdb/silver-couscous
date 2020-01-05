# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 21:01:25 2019

@author: Timot
"""


import networkx as nx  
import matplotlib.pyplot as plt
import numpy as np
import random
#import math


class Agent:
    

    
    
    agentID = 0
    
    def __init__(self, graph):
        self.id = Agent.agentID #should tie this to node
        self.name = "Agent " +str(self.id)
        Agent.agentID += 1
        
        graph.add_node(self.id, agent = self)
        self.node = graph[self.id]
        self.neighbours = set()
        
        self.strategy = Strategy().randomize()
        self.wallet = []        # wallet keeps track of total revenue #array will be more efficient later
        self.revenue = []       # revenue just keeps track of gains per round
        self.bestNeighbour = 0
    
    def __repr__(self):
        return("Agent %d" %self.id)# + "S({0},{1})".format(self.strategy['offer'], self.strategy['accept']))
        
    
    def getStrategy(self):
        return(self.strategy)
    
    
    def introduce(self):
        print("Agent {0} born with strategy {1}".format(self.id, self.strategy))
            
        
    def budgeting(self, payoff):
        self.revenue.append(payoff)
        
        
    def adapt(self, graph):                                    # first check highest value, after update strategy
        
        self.wallet.append(round(np.mean(self.revenue), 2))
        payMax = [np.sum(self.revenue), self]                   # calc with sum or mean?
        
        self.neighbours = graph.neighbors(self.id)
        
        if testing:
            print("\n{0}:".format(self))
          
        
        for n in self.neighbours:
            
            neighRevenue = np.sum(graph.nodes[n]['agent'].revenue)
            
            if testing:
                print("{0} : revenue = {1}".format(graph.nodes[n]['agent'], round(neighRevenue, 2)))
            
            if payMax[0] < neighRevenue:
                
                payMax = [neighRevenue, graph.nodes[n]['agent']]
            
            # if neighbours have similar revenue, none can be benefited by neighbour order. doesn't account for focal agent
            
            if payMax[0] is neighRevenue and payMax[1] != self:
                choiceList = [payMax[0], neighRevenue]
                payMax[0] = random.choice(choiceList)
                
        
        # bestNeighbour is the neighbour that will be imitated
        self.bestNeighbour = payMax[1]
        
        if testing:
            if payMax[0] > np.mean(self.revenue):
                print("bestNeighbour of {0} is {1} with {2} over {3}".format(self, self.bestNeighbour, payMax[0], np.sum(self.revenue))) #mean or sum?
                

    def updateStrategy(self, currentRound):
        self.revenue.clear()
        
        if testing:
            print("{0}: Revenue of best neighbour {1}: {2}".format(self, self.bestNeighbour, np.mean(self.bestNeighbour.wallet[currentRound])))
        
        #insert probability here
        revSelf = self.wallet[currentRound]
        revOpp = self.bestNeighbour.wallet[currentRound]
        
        changeProb = (revOpp - revSelf) / max(len(self.node), len(self.bestNeighbour.node))
        #print("this is changeprob: {0}, len({3}):{1}, len({4}):{2}".format(round(changeProb, 2), len(self.node), len(self.bestNeighbour.node), self, self.bestNeighbour))
        
        if random.random() < explore:
            self.strategy['offer'] = random.randint(1,10)/10
            self.strategy['accept'] = random.randint(1,10)/10
            print("{0} exploring and taking new strategy {1}".format(self, self.strategy))
            
        elif proportional:
            if random.random() < changeProb: 
                self.strategy = self.bestNeighbour.strategy
        else:
            self.strategy = self.bestNeighbour.strategy
            print("{0} exploiting strategy from {1}: {2}".format(self, self.bestNeighbour, self.strategy))
      
        
    """
    def revenueClear(self):
        self.revenue.clear()
    """        
    def shareStats(self):
        stats = [self.strategy['offer'], self.strategy['accept'], np.mean(self.revenue)]
        return(stats)
        
        
class Population:
    
    
    def __init__(self, agentCount, edgeDegree):
        self.agentCount = agentCount
        self.agents = set()
        self.graph = nx.Graph()
        self.degree = edgeDegree
        
        
    def populate(self):
        birth = 0
        while birth < self.agentCount:
            agent = Agent(self.graph)
            agent.introduce()
            self.agents.add(agent)
            birth += 1
    
    
    def constructGraph(self):
        for agent in self.agents:   #can be improved maybe?
            
            while len(agent.node) < self.degree:
                #candidates = [c for c in self.agents if len(c.node) < self.degree and c != agent]
                candidates = [c for c in self.agents if c != agent]

                for neighbour in random.sample(candidates, (self.degree - len(agent.node))):
                    if neighbour is agent:
                        continue
                    else:
                        self.graph.add_edge(agent.id, neighbour.id)
                        
        for agent in self.agents:
            print("this is the amount of neighbours for agent {0}: {1}".format(agent.id, len(agent.node)))
        nx.draw(self.graph, node_color='r', with_labels=True, alpha=0.53, width=1.5)
        plt.show()


 
           
class ultimatumGame:
    
    
    def __init__(self, rounds, agentCount, degree):
        self.population = Population(agentCount, degree)
        self.rounds = rounds
        
        self.data = np.zeros((self.rounds, 6), dtype=float)
        self.offerList = []
        self.acceptList = []
        self.payList = []
        
        self.plotting = Plot(self.rounds, self.data)

        
        if degree >= agentCount:
            raise ValueError("Amount of edges per node cannot be equal to or greater than amount of agents")
            
            
    def game(self, proposer, responder):
        if testing:
            print("Ultimatum Game between (proposer) {0} and (responder) {1}:"
                  .format(proposer, responder))
            
        offer = proposer.getStrategy()['offer']
        accept = responder.getStrategy()['accept']
        
        if offer >= accept:
            payPro = round(1 - offer, 1)
            payRes = offer   
            if testing:
                print("Payoffs for proposer {0} = {1} and for responder {2} = {3}"
                      .format(proposer, payPro, responder, payRes))
        
        else:
            payPro = 0
            payRes = 0
            if testing:
                print("Offer {0} ({1}) too low for acceptance {2} ({3})"
                      .format(offer, proposer, accept, responder))
    
        proposer.budgeting(payPro)
        responder.budgeting(payRes)
                
    
    def play(self):
        self.population.populate()
        self.population.constructGraph()
        
        struct = self.population.graph
        
        for n in range(self.rounds):
            print("\n=== round {0} ===".format(n))
            
            #proposers = set(random.sample(self.population.agents, math.trunc(agentCount/2)))
            #responders = self.population.agents - proposers
            
            for edge in struct.edges:
                proposers = random.sample(edge, 2)
                responders = proposers[::-1]

                if randomPlay:
                    proposer = struct.nodes[proposers[0]]['agent']
                    responder = struct.nodes[responders[0]]['agent']
                    
                    # ensures that agents play similar amount of games as with non-random play
                    for i in range(2):                      
                        self.game(proposer, responder)
                else:
                    if testing:
                        print(proposers)
                    for i in range(len(proposers)):
                        proposer = struct.nodes[proposers[i]]['agent']
                        responder = struct.nodes[responders[i]]['agent']
                        self.game(proposer, responder)
                        

            for agent in self.population.agents:
                agent.adapt(struct)
            print("\n")
            
            for agent in self.population.agents:
                stats = agent.shareStats()
                
                self.offerList.append(stats[0])
                self.acceptList.append(stats[1])
                self.payList.append(stats[2])
                
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
                print("final payoff for {0} is: {1}, average: {2}".format(agent, round(np.sum(agent.wallet),4), round(np.mean(agent.wallet), 4)))
        
        
        #self.offerList.clear()
        #self.acceptList.clear()
        #self.payList.clear()
        
        #plt.subplot(3,1,1)
        self.plotting.offerPlot()
        #plt.subplot(3,1,2)
        self.plotting.acceptPlot()
        #plt.subplot(3,1,3)
        self.plotting.payPlot()
        #plt.show()
        
class Plot:
    
    def __init__(self, rounds, stats):
        self.name = "None"
        self.rounds = rounds
        self.stats = stats
        
            
    def offerPlot(self):
        
        xval = range(self.rounds)
        yval = self.stats[0:self.rounds,0]
        err = self.stats[0:self.rounds,1]
        
        self.doPlot(xval, yval, err, "Average offer per round")
        """
        plt.plot(range(self.rounds), self.offersM)
        plt.title("Average offer per round")
        plt.fill_between(range(self.rounds), self.offersM + self.offersV, self.offersM - self.offersV)
        plt.ylim([0,1])
        plt.show()
        """
    def acceptPlot(self):
        xval = range(self.rounds)
        yval = self.stats[0:self.rounds,2]
        err = self.stats[0:self.rounds,3]
        
        self.doPlot(xval, yval, err, "Average accept per round")
        
        """
        plt.plot(range(self.rounds), self.acceptsM)
        plt.title("Average accept per round")
        plt.ylim([0,1])
        plt.show()
        """
    def payPlot(self):
        xval = range(self.rounds)
        yval = self.stats[0:self.rounds,4]
        err = self.stats[0:self.rounds,5]
        
        self.doPlot(xval, yval, err, "Average payoff per round")
        
    def doPlot(self, x, y, err, name):
        
        #xdata = np.arange(self.rounds)
        #ydata = np.empty(self.rounds, dtype=float)
        #error = np.empty(self.rounds, dtype=float)
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
        strategy["offer"] = random.randint(1,9)/10
        strategy["accept"] = random.randint(1,9)/10
        return(strategy)
    

    
    
class Simulation:
    
    def __init__(self):#, nSim, rounds, agentCount, edgeDegree):
        self.nSim = nSim
        self.rounds = rounds
        self.agentCount = agentCount
        self.edgeDegree = edgeDegree
        
        for sim in range(nSim):
            UG = ultimatumGame(rounds, agentCount, edgeDegree)
            UG.play()



simulations = 1
rounds = 10
agentCount = 2
edgeDegree = 1

# STOCHASTICITY
explore = 0.6       # with prob [explore], agents adapt strategy randomly. prob [1 - explore] = unconditional/proportional imit

proportional = True
randomPlay = True

testing = True

UG = ultimatumGame(rounds, agentCount, edgeDegree)
UG.play()

# ideas: graph generator for e.g. difference in SP, CC, various graphs (random, regular well-mixed etc)

#test
