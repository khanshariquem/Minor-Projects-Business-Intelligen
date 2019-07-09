from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions
from graphframes import *
from pyspark.sql.functions import explode
from operator import add
sc=SparkContext("local", "degree.py")
sqlContext = SQLContext(sc)

def closeness(g):
	
	# Get list of vertices. We'll generate all the shortest paths at
	# once using this list.
	# YOUR CODE HERE
    vertices = g.vertices.rdd.map(lambda x : (x[0])).collect()
    #print(vertices)
    shortest_paths = g.shortestPaths(vertices)
    #shortest_paths.rdd.map(lambda x: (x[0],sum_map(x[1])))
    shortest_paths  = shortest_paths.select("id", explode("distances"))
    shortest_paths = shortest_paths.rdd.map(lambda x: (x[0],x[2])).reduceByKey(add)
    shortest_paths = shortest_paths.map(lambda x: (x[0],float(1.0/float(x[1]))))
    
    df = sqlContext.createDataFrame(shortest_paths,['id','closeness'])
    
    return df

print("Reading in graph for problem 2.")
graph = sc.parallelize([('A','B'),('A','C'),('A','D'),
	('B','A'),('B','C'),('B','D'),('B','E'),
	('C','A'),('C','B'),('C','D'),('C','F'),('C','H'),
	('D','A'),('D','B'),('D','C'),('D','E'),('D','F'),('D','G'),
	('E','B'),('E','D'),('E','F'),('E','G'),
	('F','C'),('F','D'),('F','E'),('F','G'),('F','H'),
	('G','D'),('G','E'),('G','F'),
	('H','C'),('H','F'),('H','I'),
	('I','H'),('I','J'),
	('J','I')])
	
e = sqlContext.createDataFrame(graph,['src','dst'])
v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()
print("Generating GraphFrame.")
g = GraphFrame(v,e)

print("Calculating closeness.")
closeness_df = closeness(g).sort('closeness',ascending=False)
closeness_df.show()
print("Writing coloseness to file centrality_out.csv")
closeness_df.toPandas().to_csv("centrality_out.csv")
