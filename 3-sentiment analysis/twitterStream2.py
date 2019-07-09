from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import tkinter
import matplotlib
import matplotlib.pyplot as plt

def main():
    conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec
    ssc.checkpoint("checkpoint")

    pwords = load_wordlist("positive.txt")
    nwords = load_wordlist("negative.txt")
   
    counts = stream(ssc, pwords, nwords, 100)
    make_plot(counts)


def make_plot(counts):
    """
    Plot the counts for the positive and negative words for each timestep.
    Use plt.show() so that the plot will popup.
    """
    # YOUR CODE HERE
    
    pCounts = [] 
    nCounts = []
    
    for count in counts:
        pCounts.append(count[0][1])
        nCounts.append(count[1][1])
    
    plt.plot(pCounts, 'bo-', label='Positive')
    plt.plot(nCounts, 'go-', label='Negative')
#    plt.plot(counts)
    plt.xlabel("Time Step")
    plt.ylabel("Word count")
    plt.legend(loc='upper left')
    plt.show()


def load_wordlist(filename):
    """ 
    This function should return a list or set of words from the given filename.
    """
    with open(filename) as f:
        # return the split results, which is all the words in the file.
        return f.read().split()

def updateFunction(newValues, runningCount):
    if runningCount is None:
        runningCount = 0
    return sum(newValues, runningCount)  # add the new values with the previous running count to get the new count    


def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1])
    #tweets.pprint()
    # Each element of tweets will be the text of a tweet.
    # You need to find the count of all the positive and negative words in these tweets.
    # Keep track of a running total counts and print this at every time step (use the pprint function).
    # YOUR CODE HERE
    words = tweets.flatMap(lambda line: line.split(" "))
#    pairs = words.map(lambda word: [("positive", 1) if  word.strip() in pwords else ("negative", 1)])
    pairs = words.map(lambda word: ("positive", 1) if word.strip() in pwords else ("positive", 0)).union(words.map(lambda word: ("negative", 1) if word.strip() in nwords else ("negative", 0)))

    wordCounts = pairs.reduceByKey(lambda x, y: x + y)
    #wordCounts.pprint()
    
    totalCount = tweets.updateStateByKey(updateFunction)
    totalCount.pprint()
    # Let the counts variable hold the word counts for all time steps
    # You will need to use the foreachRDD function.
    # For our implementation, counts looked like:
    #   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
    counts = []
    wordCounts.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    
    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)

    return counts


if __name__=="__main__":
    main()
