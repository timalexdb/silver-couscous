# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 21:01:25 2019

@author: Timot
"""


import networkx as nx  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
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
         self.graphData = dict()
         self.char1 = list()
         self.char2 = list()
         
    def createGraph(self):
        m_step = np.linspace(1, agentCount, simulations, endpoint=False)
        p_step = np.linspace(0, 0.6, simulations)
        
        #print(plist)
        
        for i in range(simulations):
            
            p = p_step[i]
            #step = (agentCount-1) / simulations
            if int(m_step[i]) < agentCount:
                m = int(m_step[i])
            else:
                m = (agentCount-1)
            #print("M for barabasi-Albert: {0}".format(m))
            
            # --> nx.extended_barabasi_albert_graph?
            SFN = nx.barabasi_albert_graph(agentCount, m) # scale-free network characterised by having vastly differing degrees (hence scale-free), small amount of large hubs
            SWN = nx.connected_watts_strogatz_graph(agentCount, edgeDegree, p) # small-world network characterised by low 
            
            #SWN = nx.newman_watts_strogatz_graph(agentCount, edgeDegree, p)
            #while not nx.is_connected(SWN):
            #    SWN = nx.newman_watts_strogatz_graph(agentCount, edgeDegree, p)
            self.char1.append(Graph.graphCharacteristics(SWN))
            self.char2.append(Graph.graphCharacteristics(SFN))
            
            nx.write_edgelist(SWN, "Graphs/Watts-StrogatzV{0}".format(i))
            nx.write_edgelist(SFN, "Graphs/Barabasi-AlbertV{0}".format(i))

            
# =============================================================================
#            can this part directly below delet plos is for testng
# =============================================================================
            #read = open("Graphs/test_connWattStroV{0}".format(i), 'rb')
            #graph = nx.read_edgelist(read, nodetype=int)
# =============================================================================
#             'til here
# =============================================================================
            #nx.draw(graph, node_color='b', with_labels=True, alpha=0.53, width=1.5)
            #plt.show()
            #nx.draw(graph2, node_color='r', with_labels=True, alpha=0.53, width=1.5)
            #plt.show()
            
        self.graphData['Watts-Strogatz'], self.graphData['Barabasi-Albert'] = self.char1, self.char2
        
        for key in self.graphData.keys():
            if key == 'Watts-Strogatz':
                x = p_step
                xlab = 'probability of rewiring random edge'
            if key == 'Barabasi-Albert':
                x = m_step
                xlab = 'degree for each new agent'
            Plot().measurePlot(key, x, gg.graphData, xlab)

        
        #print("characteristics of Watts-Strogatz (SWN) {0}: \n{1}".format(i,Graph.graphCharacteristics(SWN)))
        #print("characteristics of Barabasi-Albert (SFN) {0}: \n{1}".format(i,Graph.graphCharacteristics(SFN)))
        print("graphs created")
        
        
        #print(graph.nodes)
        #nx.write_edgelist(graph, "Graphs/test_graph")
        
        
        
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
    # implement structural measures here! also store graph struct + information here so that graphClass can be run by itself(...)
    # (...) and OFFER graphs to program instead of necessarily being called by program.
    
    def structuralPower(g):
        SPtotal = list(range(agentCount))#np.empty(agentCount)
        groups = dict()
        
        for node in g.nodes:            
            groups[node] = set(g.neighbors(node))
            groups[node].add(node)
            if testing:
                print("groups for node {0} ({1}): {2}".format(node, len(groups[node]), groups[node]))        
                
        count = 0
        
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
                if node == 8:
                    print("\nthese are group members for node {0}: {1}".format(node, neighlist))
                    print("this is group for node {0}: {1}".format(node, groups[node]))
                    print("this is SP for {0}: {1}, {2}".format(node, SP, (sum(SP)/len(reach))))
                    print("length SP: {0}, length nbh: {1}, length reach: {2}".format(len(SP), len(groups[node]), len(reach)))
                    print("this is reach for node {0}: {1}\n".format(node, reach))
            SPtotal[node] = (sum(SP)/len(reach))
        #print("this is SPtotal: {0} \naverage: {1}".format(SPtotal, np.mean(SPtotal)))
        
        if len(SPtotal) != agentCount:
            raise ValueError("something wrong with your SP.")
        
        return((np.mean(SPtotal), SPtotal))
        
    
    def barabal(n, m, seed=None):
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
            raise nx.NetworkXError(f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}")
    
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
            repeated_nodes.extend([source] * m)
            # Now choose m unique nodes from the existing nodes
            # Pick uniformly from repeated_nodes (preferential attachment)
            targets = nx._random_subset(repeated_nodes, m, seed)
            source += 1
            
        return G

    """
    def _random_barabal(seq, m, rng, g):
        targets = set()
    
        while len(targets) < m:
            x = rng.choice(seq, p = _pref_attach(seq))
            targets.add(x)
            
        return targets
    
    def _pref_attach(seq, g):
        prefList = []
        
        for node in g.nodes():
            p = (g.degree[node]/g.size)
            print("node {0} with preference probability {1}".format(g.nodes[node], p))
            prefList.append(p)
        return prefList
    """
    
       
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
        #nx.draw(self.graph, node_color='r', with_labels=True, alpha=0.53, width=1.5)
        #plt.show()


           
class ultimatumGame:
    
    
    def __init__(self, graph):
        self.population = Population(graph)
        
        self.data = np.zeros((rounds, 6), dtype=float)
        #self.data2 = pd.DataFrame(index=range(rounds))
        self.offerList = []
        self.acceptList = []
        self.payList = []
        
        self.graph = self.population.graph
        
        self.plotting = Plot()

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
                
                # row = round, col = agent, input = all stats from each agent in that round
                data2.iloc[n, agent.id] = stats
                
                
                self.offerList.append(stats[0])
                self.acceptList.append(stats[1])
                self.payList.append(stats[2])
                
                if n != (rounds - 1):
                    agent.updateStrategy(n)
                
            self.data[n, 0], self.data[n, 1] = np.mean(self.offerList), np.var(self.offerList)
            #self.data[n, 1] = np.var(self.offerList)
            self.data[n, 2], self.data[n, 3]  = np.mean(self.acceptList), np.var(self.acceptList)
            #self.data[n, 3] = np.var(self.acceptList)
            self.data[n, 4], self.data[n, 5] = np.mean(self.payList), np.var(self.payList)
            #self.data[n, 5] = np.var(self.payList)
            
            self.offerList.clear()
            self.acceptList.clear()
            self.payList.clear()
        
        if testing:           
            for agent in self.population.agents:
                print("\ntotal wallet for {0}: {1}".format(agent, agent.wallet))
                print("final payoff for {0} is: {1}, average: {2}".format(agent, round(np.sum(agent.wallet),4), round((np.sum(agent.wallet)/agent.successes), 4)))
        
        # self.plotting.offerPlot(self.data)
        # self.plotting.acceptPlot(self.data)
        # self.plotting.payPlot(self.data)
        
    
class Plot:
    
    def __init__(self):
        self.name = "None"
                
    def offerPlot(self, stats):
        
        xval = range(rounds)
        yval = stats[0:rounds,0]
        err = stats[0:rounds,1]
        
        self.doPlot(xval, yval, err, "Average offer per round")
        
    def acceptPlot(self, stats):
        xval = range(rounds)
        yval = stats[0:rounds,2]
        err = stats[0:rounds,3]
        
        self.doPlot(xval, yval, err, "Average accept per round")
        
    def payPlot(self, stats):
        xval = range(rounds)
        yval = stats[0:rounds,4]
        err = stats[0:rounds,5]
        
        self.doPlot(xval, yval, err, "Average payoff per round")
        
    def measurePlot(self, key, xval, graphCharacteristics, xlab):
        
        #for g in graphType:
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
                  
        
        
class Strategy:
    
    def __init__(self):
        self.name = "None"
        
    def randomize(self):
        strategy = {}
        strategy["offer"] = random.choice(list(range(1,10,1)))/10
        strategy["accept"] = random.choice(list(range(1,10,1)))/10
        if strategy["offer"] > 0.9:
            raise ValueError("randomize screws up offer")
        if strategy["accept"] > 0.9:
            raise ValueError("randomize screws up accept")
        return(strategy)
    

    
    
class Simulation:
    
    def __init__(self):
        
        self.data = np.zeros((rounds, agentCount, simulations), dtype=float) # n of rounds, n of agents, n of values, amount of sims
        self.finalPlot = Plot()
        
    
    def run(self):
        graphType = ['Watts-Strogatz', 'Barabasi-Albert']
                
        for g in graphType:
            
            for sim in range(simulations):
                print("\n=== Commencing Simulation {0} ===\n".format(sim))
                
                read = open("Graphs/{0}{1}".format(g, 'V'+str(sim)), 'rb')
                #read = open("Graphs/test_connWattStroV{0}".format(sim), 'rb')
                graph = nx.read_edgelist(read, nodetype=int)
                
                #print("characteristics of {0}: \n{1}".format(g, gg.graphCharacteristics(graph)))
                
                if showGraph:
                    nx.draw(graph, node_color='r', with_labels=True, alpha=0.53, width=1.5)
                    plt.title('{0} (simulation {1}/{2})'.format(g, sim+1, simulations))
                    plt.show()
                    
                
                if len(graph) > agentCount:
                    raise ValueError("agents incorrectly replaced")
                
                print("characteristics of {0}({1}): \n{2}".format(g, sim, gg.graphData[g][sim]))
                
                
                UG = ultimatumGame(graph)
                UG.play()
                
                UG.population.killAgents()
                
                
                #arrayTest[:,:, sim] = data2
                
                #data3[sim,] = data2
                #self.data[:,:,sim] = UG.getData()
                
                """
                if 'agentInfo' not in globals():
                    agentInfo = xr.DataArray(data2)
                    agentInfo.expand_dims('sim')
                else:
                    xr.concat((agentInfo, data2), dim='sim')
                """
                if sim == 0:
                    simData = xr.DataArray(data = data2, dims = ('Rounds', 'Agent')).expand_dims('Sim')
                    
                else:
                    simData = xr.concat([simData, xr.DataArray(data = data2, dims = ('Rounds', 'Agent')).expand_dims('Sim')], dim='Sim')
            
            
            if graphType.index(g) == 0:
                gData = xr.DataArray(data = simData).expand_dims('Graph')
                
            else:
                gData = xr.concat([gData, xr.DataArray(data = simData).expand_dims('Graph')], dim='Graph')
                
        gData = gData.assign_coords({'Rounds' : range(rounds), 'Agent' : agentList, 'Sim': range(simulations), 'Graph' : graphType})#.transpose(..., 'Agent')
        
        return(gData)

# =============================================================================
# HYPERPARAM
# =============================================================================

simulations = 10
rounds = 10
agentCount = 15
edgeDegree = 4


explore = 0.4       # with prob [explore], agents adapt strategy randomly. prob [1 - explore] = unconditional/proportional imit


# =============================================================================
# GAME SETTINGS
# =============================================================================
proportional = True
randomPlay = True

testing = False
demo = False
showGraph = False

agentList = np.array(["Agent %d" % agent for agent in range(0,agentCount)])



#dataMI = pd.DataFrame(np.zeros((40)),index = datadx)#np.random.randn(len(datadx), 1), index = datadx)
data2 = pd.DataFrame(index=range(rounds), columns = agentList)       # just been messing around with this. REMEMBER .ILOC()


#pd.DataFrame(index=range(simulations)) #can't get Data2 into Data3.

gg = Graph()
gg.createGraph()

game = Simulation()
blaTest = game.run()#.to_dataframe('Agent') # xr.DataArray(data=game.run(), coords={'Rounds' : range(rounds), 'Agent' : agentList, 'Sim': range(simulations)})

# ideas: graph generator for e.g. difference in SP, CC, various graphs (random, regular well-mixed etc)

