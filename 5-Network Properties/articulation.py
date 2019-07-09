import sys
import time
import networkx as nx
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions
from graphframes import *
from copy import deepcopy

sc=SparkContext("local", "degree.py")
sqlContext = SQLContext(sc)

def articulations(g, usegraphframe=False):
	# Get the starting count of connected components
	# YOUR CODE HERE
    	new_df=''
	# Default version sparkifies the connected components process 
	# and serializes node iteration.
	if usegraphframe:
		# Get vertex list for serial iteration
		# YOUR CODE HERE
		#g.connectedComponents().show()
		#g.connectedComponents().groupby(['component']).show()
        	nof_compon = g.connectedComponents().select('component').distinct().count()
		#print(nof_compon)
		count_dict = []
		vertices = g.vertices.rdd.map(lambda x : x[0]).collect()
		for vertex in vertices:
            		new_vertices = g.vertices.filter('id !="' + vertex + '"' )
			new_edges = g.edges.filter('src != "'+ vertex +'"').filter('dst != "'+ vertex +'"')
			#new_vertices = sqlContext.createDataFrame(sc.parallelize(new_vertices),['id'])
            		new_graph = GraphFrame(new_vertices,new_edges)
            		new_nof_compon = new_graph.connectedComponents().select('component').distinct().count()
			#print(new_nof_compon)
            		if new_nof_compon>nof_compon:
                		#count_dict[vertex] = 1
				count_dict.append((str(vertex),1))
            		else:
                		#count_dict[vertex] = 0
				count_dict.append((str(vertex),0))
        	#print(count_dict)
        	new_df = sqlContext.createDataFrame(sc.parallelize(count_dict),['id','articulation']) 
		# For each vertex, generate a new graphframe missing that vertex
		# and calculate connected component count. Then append count to
		# the output
		# YOUR CODE HERE
		
	# Non-default version sparkifies node iteration and uses networkx 
	# for connected components count.
	else:
        # YOUR CODE HERE
		network_graph = nx.Graph()
		vertices = g.vertices.rdd.map(lambda x : x[0]).collect()
		edges = g.edges.rdd.map(lambda x : (x[0],x[1])).collect()
		network_graph.add_nodes_from(vertices)
		network_graph.add_edges_from(edges)
		nof_compon = nx.number_connected_components(network_graph)
		#print(nof_compon)
		count_dict = []	
		for vertex in vertices:
			new_graph = deepcopy(network_graph)
			new_graph.remove_node(vertex)
			new_nof_compon = nx.number_connected_components(new_graph)
			if new_nof_compon>nof_compon:
                		#count_dict[vertex] = 1
				count_dict.append((str(vertex),1))
            		else:
                		#count_dict[vertex] = 0
				count_dict.append((str(vertex),0))
		#print(count_dict)
		new_df = sqlContext.createDataFrame(sc.parallelize(count_dict),['id','articulation']) 
		
	return (new_df)


filename = sys.argv[1]
lines = sc.textFile(filename)

pairs = lines.map(lambda s: s.split(","))
e = sqlContext.createDataFrame(pairs,['src','dst'])
e = e.unionAll(e.selectExpr('src as dst','dst as src')).distinct() # Ensure undirectedness 	

# Extract all endpoints from input file and make a single column frame.
v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()	

# Create graphframe from the vertices and edges.
g = GraphFrame(v,e)

#Runtime approximately 5 minutes
print("---------------------------")
print("Processing graph using Spark iteration over nodes and serial (networkx) connectedness calculations")
init = time.time()
df = articulations(g, False)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)

print("Writing artcultaion fetched using serial (networkx) connectedness calculations to file articulations_out.csv")
df.toPandas().to_csv("articulations_out.csv")
print("---------------------------")

#Runtime for below is more than 2 hours
print("Processing graph using serial iteration over nodes and GraphFrame connectedness calculations")
init = time.time()
df = articulations(g, True)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)
