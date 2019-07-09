import pandas as pd
from scipy import spatial as sp
import copy as cp
import sys
import numpy as np
import igraph as ig




#initialization
if (len(sys.argv) != 2):
  print("Enter Alpha")
  sys.exit(1)
else:
  alpha = float(sys.argv[1])    
gn = pd.read_csv('./data/fb_caltech_small_attrlist.csv')
graph_edges = []
filename = open('./data/fb_caltech_small_edgelist.txt')    
for l in filename:
  graph_edges.append(tuple(map(int,  l.split(" "))))
mygraph = ig.Graph()
mygraph.add_vertices(len(gn))
mygraph.add_edges(graph_edges)
k=gn.keys()
for topic in k:
  mygraph.vs[topic]=gn[topic]


def membership(x,clus):
  for y in clus:
    if x in y:
      return y
  return []

def deltanew(mygraph, v , clus):
  sum_Gix,sum_di,m,dx = 0,0,mygraph.ecount(),mygraph.degree(v)
  for cl in cluster:
    if mygraph.are_connected(v, cl):
        sum_Gix += 1
    sum_di += mygraph.degree(cl)
  return  1/(2*m)*(sum_Gix - (dx/(2*m))*sum_di)


def deltatt(v , clus , sim):
  md=0
  for cluster in clus:
    md=md+sim[v][cluster]
  return md/len(clus)



def modgain(mygraph, n , clus , similarity_matrix , alpha):
  val1=alpha*deltanew(mygraph,n,clus)
  val2=(1-alpha)*deltatt(n,clus,similarity_matrix)
  return val1+val2



def modcalc(graph, cluster):
    member = np.zeros((graph.vcount(), ), dtype = int)
    for idx, c in enumerate(cluster):
        for vertex in c:
            member[vertex] = idx
    return graph.modularity(member, weights=None)


#Phase1
#Creating Similarity Matrix 


def similaritymatrixcal(x, y, g):
  attri1=list(g.vs[x].attributes().values())
  attri2=list(g.vs[y].attributes().values())
  dist=1-sp.distance.cosine(attri1,attri2)
  return dist 

similarmatrix = np.zeros((mygraph.vcount(),mygraph.vcount()))
clusters = [[x] for x in range(mygraph.vcount() )]
for i in range(mygraph.vcount() ):
  for j in range(mygraph.vcount() ):
    similarmatrix[i][j] = similaritymatrixcal(i, j, mygraph)
for i in range(15):
  for vertex in range(mygraph.vcount()):
    max_gain = float("-inf")
    curr_cluster =membership(vertex,clusters)
    new_cluster = []
    for cluster in clusters:
      if set(curr_cluster) != set(cluster):
        gain = modgain(mygraph,vertex,cluster,similarmatrix,alpha)
        if gain and gain > max_gain:
          max_gain,new_cluster= gain,cluster
    if max_gain>0:
      new_cluster.append(vertex)
      curr_cluster.remove(vertex)
      if not len(curr_cluster) :
        clusters.remove(curr_cluster)
 
oldmod = modcalc(mygraph,clusters)
new_vertex = [0 for x in range(mygraph.vcount() )]
oldict_cluster = {}
for index, clus in enumerate(clusters):
    for vertex in clus:
        new_vertex[vertex] = index
    oldict_cluster[index] = clus
mygraph.contract_vertices(new_vertex, combine_attrs="sum")
phase1clus = cp.deepcopy(clusters)

#Phase2
#Creating Similarity Matrix again based on New vertex

  
clusters = [[c] for c in range(mygraph.vcount() )]
for i in range(15):
    for vertex in range(mygraph.vcount() ):
        curr_cluster = membership(vertex,clusters)
        new_cluster = []
        max_gain =float("-inf")
        for cluster in clusters:
            if set(curr_cluster) != set(cluster):
                gain = modgain(mygraph,vertex,cluster,similarmatrix,alpha)
                if gain>0 and gain > max_gain:
                    max_gain,new_cluster= gain,cluster
        if max_gain>0:
          new_cluster.append(vertex)
          curr_cluster.remove(vertex)
          if not len(curr_cluster) :
            clusters.remove(curr_cluster)
 
   
newmod = modcalc(mygraph,clusters)
phase2clus = []
for cluster in clusters:
    newlist = []
    for c in cluster:
        newlist.extend(oldict_cluster[c])
    phase2clus.append(newlist)
 
print(len(phase1clus))
print(len(phase2clus))
def writer(clus, alpha):
  if alpha==0.5:
    a=5
  else:
    a=int(alpha)
  f="communities_"+str(a)+".txt"
  ff = open(f, 'w+')
  for m in clus:
    for i in range(len(m)-1):
      ff.write(str(m[i])+",")
    ff.write(str(m[-1])+"\n")    
  ff.close()

print(newmod)
print(oldmod)

if(newmod < oldmod):
    writer(phase1clus,alpha)
else:
    writer(phase2clus,alpha)