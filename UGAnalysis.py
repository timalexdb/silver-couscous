# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:54:14 2020

@author: Timot
"""
import networkx as nx  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import random

graphType = ['Watts-Strogatz', 'Barabasi-Albert']

set_self = True

if set_self:
    simulations = 5
    rounds = 5
    agentCount = 14
    edgeDegree = 4
    
    explore = 0.4
    
    proportional = True
    randomPlay = True
#else:
#    simulations, rounds, agentCount, edgeDegree, explore, proportional, randomPlay = settings

for g in graphType:
    exec('finalDat_{0} = pd.read_csv("Data/gameData_n{1}_sim{2}_round{3}_exp={4:.2f}_prop={5}_random={6}_{7}.csv" , encoding="utf-8", header = [0,1])'.format(str(g)[0], agentCount, simulations, rounds, explore, str(proportional), str(randomPlay), g))

