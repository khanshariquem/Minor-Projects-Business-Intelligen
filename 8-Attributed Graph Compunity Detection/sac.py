import pandas as pd
import numpy as np
import sys
from igraph import *
from scipy import spatial

def firstphase(g,a,c,l1,l2,sm):
    i,checker=0,False
    while (checker != True and i<15):
        checker=True
        for v1 in range(0,l1):
            maxv=-1
            maxdel=float(0)
            cluster=[set(c)]
            for v2 in cluster:
                if (c[v1]!=v2):
                    qnew1=g.modularity(c, weights='weight')
                    t,c[v1]=c[v1],v2
                    qnew2=g.modularity(c, weights='weight')
                    c[v1]=t
                    qnew=qnew2-qnew1
                    mul1=a*qnew
                    q1new=float(0)
                    countind=0
                    ind=[]
                    for i,x in enumerate(c):
                        if(x==v2):
                            ind.append(i)
                            countind+=1
                    for i in ind:
                        q1new+=sm[v1][i]
                    dist=len(set(c))
                    q2new=q1new/(dist*countind)
                    mul2=(1-a)*q2new
                    final=mul1+mul2
                    if(final>maxdel):
                        maxv=v2
                        maxdel=final
            if(maxdel!=0 and maxv!= -1):
                checker=False
                c[v1]=maxv
        i+=1
    return c
                    
                        
def seqCluster(C,ma,newlist,val):
    for x in C:
        if x not in ma:
            newlist.append(val)
            ma[x]=val
            val+=1
        else:
            newlist.append(ma[x])
    return newlist
                    

def DeltaQAttr(C, g, v1, v2):
	S = 0.0;
	indices = [i for i, x in enumerate(C) if x == v2]
	for v in indices:
		S = S + simMatrix[v1][v]
	return S/(len(indices)*len(set(C)))

def QAttr(C,g,sm):
	clusters=[Clustering(C)]
	S=float(0)
	V=g.vcount()
	count=0
	for c in clusters:
		T=float(0)
		count+=1
		for v1 in c:
			for v2 in C:
				if (v1!=v2):
					T+=sm[v1][v2]
		T=T/count
		S+=T
	val=S/(len(set(C)))
	return val

def createsim(g,n):
    a=[]
    for l in range(0,n):
            a.append([0] * n)
    for i in range(0,n):
        for j in range(0,n):
            var1=g.vs[v1].forgraph().values()
            var2=g.vs[v2].forgraph().values()
            cos=spatial.distance.cosine(var1,var2)
            val=1-cos
            a[i][j]=val
    b=a.copy()
    return a,b

def writeToFile(clusters, a):
	file = open("communities_"+a+".txt", 'w+')
	for c in clusters:
		for i in range(len(c)-1):
			file.write("%s," % c[i])
		file.write(str(c[-1]))
		file.write('\n')
	file.close()

def phase2 (g,C,sm1,sm2):
    nc=seqCluster(C,{},[],0)
    temp=list(Clustering(nc))
    n=len(set(nc))
    a=[]
    for l in range(0,n):
            a.append([0] * n)
	
    for i in range(n):
	    for j in range(n):
		    sim=float(0)
		    for k in temp[i]:
			    for l in temp[j]:
				    sim+=sm2[k][l]
		    a[i][j]=sim
	
    g.contract_vertices(nc)
    g.simplify(combine_edges=sum)
    return a,sm2

    


def main():

    if len(sys.argv) != 2:
        print ("Plz enter the alpha value")
        exit()
    alpha=float(sys.argv[1])
    if(sys.argv[1]!=0.5):
        ver=sys.argv[1]
    else:
        ver=int(sys.argv[1]*10)
        
    
    with open('data/data/fb_caltech_small_edgelist.txt') as f:

        edge = f.readlines()
    #print(edges)
    #it is in form of '0 2\n', '0 14\n',etc
    pairs=[]
    numpair=0
    pairs = [tuple([int(x) for x in line.strip().split(" ")]) for line in edge]
    '''
    for i in edge:
        pair=(tuple[int(x) for x in i.split(" ")])
        print(pair)
        numpair=numpair+1
        pairs.append(pair)
    #print(pairs)
    '''    
    forgraph=pd.read_csv('data/data/fb_caltech_small_attrlist.csv')
    

    #https://igraph.org/python/doc/tutorial/tutorial.html to get details about the following
    g=Graph()
    g.add_vertices(len(forgraph))
    g.add_edges(pairs)
    for i in forgraph.keys():
        g.vs[i]=forgraph[i]
    g.es['weight']=[1]*numpair

    sm1,sm2=createsim(g,len(forgraph))
    V=g.vcount()
    lenv=len(g.vs)
    leng=len(g.es)
    C=firstphase(g, alpha, range(V),lenv,leng,sm)
    print('Number of Communities after Phase 1')
    print(len(set(C)))
    d=dict()
    newlist=list()
    C = seqCluster(C,d,newlist,0)
    module1=g.modularity(C,weights='weight') + QAttr(C,g,sm)
    sm1,sm2=phase2(g,C,sm1,sm2)

    V=g.vcount()
    lenv=len(g.vs)
    leng=len(g.es)
    C2=firstphase(g, alpha, range(V),lenv,leng,sm)
    d=dict()
    newlist=list()
    newC2=seqCluster(C2,d,newlist,0)
    cPhase2 = list(Clustering(newC2))
    module2=g.modularity(C,weights='weight') + QAttr(C,g,sm)

    d=dict()
    newlist=list()
    newC1=seqCluster(C,d,newlist,0)
    Cfinally=list()
    cPhase1=list(Clustering(C1new))

    for i in cPhase2:
        t=list()
        for j in i:
            t.extend(cPhase[j])
            #https://stackoverflow.com/questions/252703/difference-between-append-vs-extend-list-methods-in-python

        Cfinally.append(t)

    if(module1<module2):
        print('Phase 2 clusters have higher modularity')
        writeToFile(cPhase2,str(ver))
	
	#return cPhase2
    else:
        print('Phase 1 clusters have higher modularity')
        writeToFile(cPhase1, str(ver))
	
	#return cPhase1


if __name__ == "__main__":
    main()
