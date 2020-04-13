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
        
        #if str(self) in dataSet.columns:
        #    del dataSet[str(self)]
        #dataSet[str(self)] = ""
    
    
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
            sys.exit("fitness no bueno chef")
                
        
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
            sys.exit("revhigher")
        
        if random.random() < changeProb:
            self.nextstrat = model.strategy
            if verbose:
                print("{0} switch!".format(self))
            #sys.exit()
            
            
    def fermi(self, currentRound):
        model = random.choice(self.neighbours)
        #fitSelf = self.wallet[currentRound] / len(self.neighbours)#/(currentRound+1)#sum(self.successes)
        #fitMod = model.wallet[currentRound] / len(model.neighbours)#/(currentRound+1)#sum(model.successes)
                
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
         self.char1 = list()
         self.char2 = list()
         self.charT = list()
         
    def createGraph(self):
        if testCase:
            
            for i in range(simulations):
                testGraph = nx.complete_graph(agentCount)
                self.charT.append(Graph.graphCharacteristics(testGraph))
                self.graphData['testCase'] = self.charT

                nx.write_gpickle(testGraph, "Graphs/testCaseV{0}_n{1}_sim{2}_round{3}_exp={4:.2f}_random={5}.gpickle".format(i, agentCount, simulations, rounds, explore, str(randomRoles)))
                
        else:
            m_step = np.linspace(1, agentCount, simulations, endpoint=False)
            p_step = np.linspace(0, 0.6, simulations)
            
            for i in range(simulations):
                
                p = p_step[i]
    
                if int(m_step[i]) < agentCount:
                    m = int(m_step[i])
                else:
                    m = (agentCount-1)
                
                # --> nx.extended_barabasi_albert_graph?
                SFN = nx.barabasi_albert_graph(agentCount, m) # scale-free network characterised by having vastly differing degrees (hence scale-free), small amount of large hubs
                SWN = nx.connected_watts_strogatz_graph(agentCount, edgeDegree, p) # small-world network characterised by low 
                
                
                self.char1.append(Graph.graphCharacteristics(SWN))
                self.char2.append(Graph.graphCharacteristics(SFN))
                
                #nx.write_edgelist(SWN, "Graphs/Watts-StrogatzV{0}".format(i))
                #nx.write_edgelist(SFN, "Graphs/Barabasi-AlbertV{0}".format(i))
                #nx.write_edgelist(SWN, "Graphs/Watts-StrogatzV{0}_n{1}_sim{2}_round{3}_exp={4:.2f}_prop={5}_random={6}".format(i, agentCount, simulations, rounds, explore, str(randomRoles)))
                #nx.write_edgelist(SFN, "Graphs/Barabasi-AlbertV{0}_n{1}_sim{2}_round{3}_exp={4:.2f}_prop={5}_random={6}".format(i, agentCount, simulations, rounds, explore, str(randomRoles)))
                nx.write_gpickle(SWN, "Graphs/Watts-StrogatzV{0}_n{1}_sim{2}_round{3}_exp={4:.2f}_random={5}.gpickle".format(i, agentCount, simulations, rounds, explore, str(randomRoles)))
                nx.write_gpickle(SFN, "Graphs/Barabasi-AlbertV{0}_n{1}_sim{2}_round{3}_exp={4:.2f}_random={5}.gpickle".format(i, agentCount, simulations, rounds, explore, str(randomRoles)))
                
    
                #nx.draw(graph, node_color='b', with_labels=True, alpha=0.53, width=1.5)
                #plt.show()
                #nx.draw(graph2, node_color='r', with_labels=True, alpha=0.53, width=1.5)
                #plt.show()
                
            self.graphData['Watts-Strogatz'], self.graphData['Barabasi-Albert'] = self.char1, self.char2
            
            if showGraph:
                for key in self.graphData.keys():
                    if key == 'Watts-Strogatz':
                        x = p_step
                        xlab = 'probability of rewiring random edge'
                    if key == 'Barabasi-Albert':
                        x = m_step
                        xlab = 'degree for each new agent'
                    Plot().measurePlot(key, x, gg.graphData, xlab)

        print("graphs created")
        
        
    def graphCharacteristics(g):
        #connectivity = nx.all_pairs_node_connectivity(g)
        APL = nx.average_shortest_path_length(g)    # average of all shortest paths between any node couple
        clust = nx.average_clustering(g)
        SPgraph, SPtotal = Graph.structuralPower(g)
        SPtotal = [round(SP, 4) for SP in SPtotal]
        
        # but which nodes??? adjust!
        charList = {'APL' : APL, 'CC' : clust, 'SPavg' : SPgraph, 'SPnodes' : SPtotal}
        # for deg dist see https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_degree_histogram.html
        return charList
        
        #   assortativity (simil of conn in the graph w.r.t. degree or other attr)
        #   average neighbor degree (average size of neighbor-neighborhood for each neighbor j of agent i)
        #   

    def structuralPower(g):
        SPtotal = list(range(agentCount))#np.empty(agentCount)
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
            print(node)
            SPtotal[node] = (sum(SP)/len(reach))
        
        if len(SPtotal) != agentCount:
            raise ValueError("something wrong with your SP.")
        
        return((np.mean(SPtotal), SPtotal))
        

    
       
class Population:
    
    def __init__(self, graph):
        
        self.agents = []#set()
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
        self.data = np.zeros((rounds, 6), dtype=float)        
        self.graph = self.population.graph           
        if edgeDegree >= agentCount:
            raise ValueError("Amount of edges per node cannot be equal to or greater than amount of agents")
    
    
    def getData(self):
        return(self.data)        
    
    
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
            #if testing:
            #print("Payoffs for proposer {0} = {1:0.2f} and for responder {2} = {3:0.2f}".format(proposer, payPro, responder, payRes))
        
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
        datalist = []
        edgedata = []
                    
        for n in range(rounds):
            print("\n=== round {0} ===".format(n+1))
            
            for edge in struct.edges:
                proposers = random.sample(edge, 2)
                responders = proposers[::-1]
                
                self.graph.edges[struct.nodes[edge[0]]['agent'].id, struct.nodes[edge[1]]['agent'].id]['round {0}'.format(n)] = []
                #self.openEdge(struct.nodes[edge[0]]['agent'], struct.nodes[edge[1]]['agent'], n)
                
                if randomRoles:
                    # ensures that agents play similar amount of games as with non-random play                    
                    for i in range(2):
                        j = random.choice(range(2))
                        proposer = struct.nodes[proposers[j]]['agent']
                        responder = struct.nodes[responders[j]]['agent']
                        self.game(proposer, responder, n)
                        #print("proposer {0} and responder {1}".format(proposer, responder))
                else:
                    
                    for i in range(len(proposers)):
                        proposer = struct.nodes[proposers[i]]['agent']
                        responder = struct.nodes[responders[i]]['agent']
                        self.game(proposer, responder, n)
                        
            for agent in self.population.agents:
                agent.storeMoney()
            
            if n != (rounds - 1):
                self.updateAgents(n)
       
        # ==== end of rounds =====
        
        for agent in self.population.agents:
            datalist.append(agent.shareData())
        
        for edge in struct.edges:
            edgedata.append(list(struct.get_edge_data(*edge).values()))        
        
        return(datalist, edgedata)
    
    
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
        
        gameTemp = []
        edgeTemp = []
        edgeList = []
        
        for g in ['Watts-Strogatz']:#graphType:
            
            for sim in range(simulations):
            
                print("\n=== Commencing Simulation {0} ===\n".format(sim))
                
                #read = open("Graphs/{0}V{1}_n{2}_sim{3}_round{4}_exp={5:.2f}_random={6}.gpickle".format(g, sim, agentCount, simulations, rounds, explore, str(randomRoles)), 'rb')
                read = open("Graphs/{0}V0_n{2}_sim{3}_round{4}_exp={5:.2f}_random={6}.gpickle".format(g, sim, agentCount, simulations, rounds, explore, str(randomRoles)), 'rb')
                graph = nx.read_gpickle(read)
                
                
                if showGraph:
                    nx.draw(graph, node_color='r', with_labels=True, alpha=0.53, width=1.5)
                    plt.title('{0} (simulation {1}/{2})'.format(g, sim+1, simulations))
                    plt.show()
                
                if len(graph) != agentCount:
                    raise ValueError("agents incorrectly replaced")
                
                print("characteristics of {0}({1}): \n{2}".format(g, sim, gg.graphData[g][sim]))
                
                UG = ultimatumGame(graph)
                simData, edgeDat = UG.play(sim)
                gameTemp.append(simData)
                edgeTemp.append(edgeDat)                
                
                #nx.write_gpickle(graph, "Graphs/{0}V{1}_n{2}_sim{3}_round{4}_exp={5:.2f}_random={6}_select={7}_beta={8}.gpickle".format(g, sim, agentCount, simulations, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity))
                
                with open("Data/{0}V{1}_n{2}_sim{3}_round{4}_exp={5:.2f}_random={6}_select={7}_beta={8}.pickle".format(g, sim, agentCount, simulations, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity), 'wb') as f:
                    pickle.dump(UG.population.agents, f)
    
                UG.population.killAgents()
                
                for edge in list(graph.edges):
                    edgeList.append((g, sim, str(edge)))
                    
                        
        indexGame = pd.MultiIndex.from_product((graphType, range(simulations), agentList), names=['Graph', 'Simulation', 'Agent']) #graphType, range(simulations), agentList), names=['Graph', 'Simulation', 'Agent'])
        indexEdge = pd.MultiIndex.from_tuples(edgeList, names=['Graph', 'Simulation', 'Edge'])
        
        gameTemp = list(map(list, zip(*(itertools.chain.from_iterable(gameTemp)))))
        edgeTemp = list(map(list, zip(*(itertools.chain.from_iterable(edgeTemp)))))
        
        gameData = pd.DataFrame(data=gameTemp, index = range(rounds), columns = indexGame)
        edgeData = pd.DataFrame(data=edgeTemp, index = range(rounds), columns = indexEdge)

        gameData.to_csv("Data/gameData_n{0}_sim{1}_round{2}_exp={3:.2f}_random={4}_select={5}_beta={6}_updating={7}_updateN={8}.csv".format(agentCount, simulations, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN), encoding='utf-8')
        edgeData.to_csv("Data/edgeData_n{0}_sim{1}_round{2}_exp={3:.2f}_random={4}_select={5}_beta={6}_updating={7}_updateN={8}.csv".format(agentCount, simulations, rounds, explore, str(randomRoles), selectionStyle, selectionIntensity, updating, updateN), encoding='utf-8')
        
        if testCase:
            print("THIS WAS A TESTCASE. N = {0}, K = {1}, G = {2}".format(agentCount, edgeDegree, graphType))


  
if __name__ == '__main__':
    # =============================================================================
    # HYPERPARAM
    # =============================================================================
    
    simulations = 100#4
    rounds = 1000#20
    agentCount = 5
    edgeDegree = 3
    
    selectionStyle = "Fermi"      # 0: unconditional, 1: proportional, 2: Fermi
    selectionIntensity = 10 # the bèta in the Fermi-equation
    
    explore = 0.02       # with prob [explore], agents adapt strategy randomly. prob [1 - explore] = unconditional/proportional imit
    
    testCase = False
    
    if testCase:
        agentCount = 2
        edgeDegree = 1

    # =============================================================================
    # GAME SETTINGS
    # =============================================================================
    
    #proportional = True
    randomRoles = False     # if false, focal agent is assigned role randomly
    
    #mutation_e = 0.2
    noise = True        # noise implemented as [strategy to exploit] ± noise_e 
    noise_e = 0.05
    
    updating = 1            # 0 : all agents update; 1 : at random (n) agents update
    updateN = 1
    
    testing = False
    showGraph = False
    
    verbose = False
    #for exp in np.arange(0.01, 0.08, step=0.01):
#    for updateAmount in range(1, 8, 2):
    
#        updateN = updateAmount
    
    gg = Graph()
    gg.createGraph()
    
    for selectionStyle in ['unconditional', 'proportional', 'Fermi']:
       
       agentList = np.array(["Agent %d" % agent for agent in range(0,agentCount)])
       
       
       graphType = ['Watts-Strogatz']#, 'Barabasi-Albert']
       
       if testCase:
           graphType = ['testCase']
       
       
       Simulation().run()    #game = Simulation().run()
                                        #game.run()

    #settings = [simulations, rounds, agentCount, edgeDegree, explore, randomRoles]
    
    # add exploration rate, proportional and randomRoles to filename
    #exec('finalDat = pd.read_csv("Data/gameData_n{0}_sim{1}_round{2}_exp={3:.2f}_prop={4}_random={5}.csv" , encoding="utf-8", header = [0,1])'.format(agentCount, simulations, rounds, explore, str(randomRoles)))
    
    #for g in graphType:
        #exec('finalDat_{0} = pd.read_csv("Data/gameData_n{1}_sim{2}_round{3}_exp={4:.2f}_prop={5}_random={6}.csv" , encoding="utf-8", header = [0,1])'.format(str(g)[0], agentCount, simulations, rounds, explore, str(randomRoles), g))

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