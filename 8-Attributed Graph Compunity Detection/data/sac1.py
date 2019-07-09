import igraph as ig
import sys
import numpy as np
import pandas as pd
from scipy import spatial as sp
import copy as cp

def writeToFile(clusters, alpha):
    if alpha == 0.5:
        filename = "communities_"+str(5)+".txt"
    elif alpha == 0.0:
        filename = "communities_"+str(0)+".txt"
    elif alpha == 1.0:
        filename = "communities_"+str(1)+".txt"
    file = open(filename, 'w+')
    for c in clusters:
        for i in range(len(c)-1):
            file.write("%s," % c[i])
        file.write(str(c[-1]))
        file.write('\n')
    file.close()

#calculate cosine similarity    
def simA(i, j, Graph):
    return 1-sp.distance.cosine(list(Graph.vs[i].attributes().values()),list(Graph.vs[j].attributes().values()))    

#find the curret cluster of a node
def current_cluster(current,all_clusters):
    for c in all_clusters:
        if current in c:
            return c
    return []

#find delta nuemann modularity gain according to equation in the paper
def calculate_delta_newmann(Graph, vertex , cluster):
#     original_modularity = Graph.modularity(all_clusters,weights = None )
#     curr_cluster.remove(vertex)
#     new_cluster.append(vertex)
#     modified_modularity = Graph.modularity(all_clusters,weights = None )
#     curr_cluster.append(vertex)
#     new_cluster.remove(vertex)
#     return (modified_modularity-original_modularity)
    sum_Gix = 0
    sum_di = 0
    for c in cluster:
        if Graph.are_connected(vertex, c):
            sum_Gix += 1
        sum_di += Graph.degree(c)
    m = Graph.ecount()
    dx = Graph.degree(vertex)
    return  1/(2*m)*(sum_Gix - (dx/(2*m))*sum_di)

#calcularty attribute modularity gain using cosine similarity
def calculate_delta_attr(vertex , cluster , similarity_matrix):
    attr_modularity = 0
    for c in cluster:
        attr_modularity += similarity_matrix[vertex][c]
    return attr_modularity/len(cluster)

#total modularity gain using nuemann modulairty , attribute modularity and given alpha
def calculate_modularity_gain(Graph, vertex , cluster , similarity_matrix , alpha):
    return alpha*calculate_delta_newmann(Graph, vertex , cluster) + (1-alpha) * calculate_delta_attr(vertex,cluster,similarity_matrix)

# Calculate the modularity of the graph after clustering
def graph_modularity(graph, cluster):
    member = np.zeros((graph.vcount(), ), dtype = int)
    for idx, c in enumerate(cluster):
        for vertex in c:
            member[vertex] = idx
    return graph.modularity(member, weights=None)

if __name__ == '__main__':
    if (len(sys.argv) != 2):
        print("Invalid Parameter")
        sys.exit(1)
    else:
        alpha = float(sys.argv[1])    

#read attributes
graph_nodes = pd.read_csv('./data/fb_caltech_small_attrlist.csv')

#read edges of graph
edges = []
file = open('./data/fb_caltech_small_edgelist.txt')    
for line in file:
    temp = line.split(" ")
    edges.append(tuple(map(int, temp)))

#initilaise the graph

Graph = ig.Graph()
Graph.add_vertices(len(graph_nodes))
Graph.add_edges(edges)
#initilaise node attributes for the graph
for col in graph_nodes.keys():
      Graph.vs[col] = graph_nodes[col]

#calculate similraity matrix of Grpah ->simA in the paper using cosine similarity        
sim_matrix = np.zeros((Graph.vcount(),Graph.vcount()))

for i in range(Graph.vcount()):
    for j in range(Graph.vcount()):
        sim_matrix[i][j] = simA(i, j, Graph)

#initailse cluster of vertices, all nodes in different clusters
#phase 1
V = Graph.vcount()  
clusters = [[c] for c in range(V)]
#print(clusters)
i = 0;
#move one cluster to another and add the same to all the clusters computing modulatity gain for each step. Compute the max gain and move the current cluster to the cluster with maximum gain if any
while i<15:
    for v in range(V):
        curr_cluster = current_cluster(v,clusters)
        new_cluster = []
        max_gain = -9999
        for cluster in clusters:
            if set(cluster) != set(curr_cluster):
                gain = calculate_modularity_gain(Graph,v,cluster,sim_matrix,alpha)
                if gain >0 and gain > max_gain:
                    max_gain = gain
                    new_cluster = cluster
        if max_gain > 0:
            curr_cluster.remove(v)
            new_cluster.append(v)
            if len(curr_cluster) == 0 :
                clusters.remove(curr_cluster)
    i = i+1;
print(clusters)


old_modularity = graph_modularity(Graph,clusters)
old_clusters = cp.deepcopy(clusters)
#merge the clusters into 1 cluster each , define new edges in graph(contract vertices) and re run phase 1 again     
#create a new dictionary of old clusters  --> old_cluster
V = Graph.vcount()
new_vertex = [0 for x in range(V)]
old_cluster = {}
for idx, c in enumerate(clusters):
    old_cluster[idx] = c
    for vertex in c:
        new_vertex[vertex] = idx
Graph.contract_vertices(new_vertex, combine_attrs="sum")


#repeat phase 1 with new vertices
V = Graph.vcount()  
clusters = [[c] for c in range(V)]
i = 0;
while i<15:
    for v in range(V):
        curr_cluster = current_cluster(v,clusters)
        new_cluster = []
        max_gain = -9999
        for cluster in clusters:
            if set(cluster) != set(curr_cluster):
                gain = calculate_modularity_gain(Graph,v,cluster,sim_matrix,alpha)
                if gain >0 and gain > max_gain:
                    max_gain = gain
                    new_cluster = cluster
        if max_gain > 0:
            curr_cluster.remove(v)
            new_cluster.append(v)
            if len(curr_cluster) == 0 :
                clusters.remove(curr_cluster)
    i = i+1;
	


#map the old clusters to new nodes, merge old and new clusters and then write
new_clusters = []
for cluster in clusters:
    templist = []
    for c in cluster:
        templist.extend(old_cluster[c])
    new_clusters.append(templist)
new_modularity = graph_modularity(Graph,clusters) 

# print(new_modularity)
# print(old_modularity)
if(new_modularity > old_modularity):
    writeToFile(new_clusters,alpha)
else:
    writeToFile(old_clusters,alpha)
