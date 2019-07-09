import pandas as pd
import numpy as np
import random
import sys


if len(sys.argv) != 2:
    print('invalid input')
    sys.exit(1)

def getOptimalMatching():
    return sum(bidsets.Budget)
    

bidsets = pd.read_csv('bidder_dataset.csv')
queries = open('queries.txt').read().split('\n')
bidsets.Budget.fillna(0,inplace=True)
OPT = getOptimalMatching()
budget = {}
temp_budget = {}
bidset={}

for rows,index in bidsets.iterrows():
    val=index['Budget']
    adv=index['Advertiser']
    keyword=index['Keyword']
    bvalue=index['Bid Value']
    if val != 0:
        budget[adv]=val
    if keyword not in bidset:
        bidset[keyword]={}
    bidset[keyword][adv]=bvalue


def getGreedyChoice(query):
    if not query:
        return 0
    bids = bidset[query]
    maxbid = 0
    maxbidder = list(bids.keys())[0]
    for advertiser,bid in bids.items():
        if  budget[advertiser] >= bid:
            if bid > maxbid:
                maxbid = bid
                maxbidder = advertiser
            elif bid == maxbid:
                if maxbidder > advertiser:
                    maxbidder = advertiser
    budget[maxbidder] -= maxbid
    return maxbid

def greedy():
    total = 0
    global temp_budget
    global budget
    for i in range(0, 100):
        random.shuffle(queries)
        temp_budget = dict(budget)
        revenue = 0;
        for query in queries:
            revenue += getGreedyChoice(query)
        #print(revenue)
        budget = dict(temp_budget)
        temp_budget = {}
        total += revenue
    return total/100

def getBalanceChoice(query):
    if not query:
        return 0
    bids = bidset[query]
    maxbalance = list(bids.keys())[0]
    for advertiser,bid in bids.items():
        if  budget[advertiser] >= bid:
            if budget[advertiser] > budget[maxbalance]:
                maxbalance = advertiser
            elif budget[advertiser] == budget[maxbalance]:
                if advertiser < maxbalance:
                    maxbalance = advertiser
    bid = bids[maxbalance]
    budget[maxbalance] -= bid
    return bid

def balance():
    total = 0
    global temp_budget
    global budget
    for i in range(0, 100):
        random.shuffle(queries)
        temp_budget = dict(budget)  
        revenue = 0;
        for query in queries:
            revenue += getBalanceChoice(query)
        budget = dict(temp_budget)
        temp_budget = {}
        total += revenue
    return total/100

def getSighValue(remaining,original):
    fraction = (original - remaining)/original
    return 1 - np.exp(fraction-1)

def getmsvvChoice(query):
    if not query:
        return 0
    bids = bidset[query]
    maxbidder = list(bids.keys())[0]
    maxvalue = 0
    for advertiser,bid in bids.items():
        if  budget[advertiser] >= bid:
            value = bid*getSighValue(budget[advertiser],temp_budget[advertiser])
            if value>maxvalue:
                maxvalue = value
                maxbidder = advertiser
            elif value==maxvalue:
                if advertiser<maxbidder:
                    maxbidder = advertiser     
    bid = bids[maxbidder]
    budget[maxbidder] -= bid
    return bid
    
def msvv():
    total = 0
    global temp_budget
    global budget
    for i in range(0, 100):
        random.shuffle(queries)
        temp_budget = dict(budget) 
        revenue = 0;
        for query in queries:
            revenue += getmsvvChoice(query)
        #print(revenue)
        budget = dict(temp_budget)
        temp_budget = {}
        total += revenue
    return total/100

def adword(algo):
    ALG = 0
    if algo == 'greedy':
        ALG = greedy()
    elif algo == 'balance':
        #pass
        ALG = balance()
    elif algo == 'msvv':
        #pass
        #print(budget)
        ALG = msvv()
    
    print(ALG)
    print(ALG/OPT)  
    
adword(sys.argv[1]) 



