from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
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
    pos = []
    neg = []
    for x in counts:
        if(len(x)==1):
            if 'negative' in x[0]:
                neg.append(x[0][1])
                pos.append(0)
            else:
                pos.append(x[0][1])
                neg.append(0)
        elif(len(x)==2):
            if 'negative' in x[0]:
                neg.append(x[0][1])
                pos.append(x[1][1])
            else:
                pos.append(x[0][1])
                neg.append(x[1][1])

    plt.plot(pos, 'ro-', label='Positive')
    plt.plot(neg, 'bo-', label='Negative')
    plt.xlabel("Time Step")
    plt.ylabel("Word count")
    plt.legend(loc='upper left')
    plt.show()


def load_wordlist(filename):
    """ 
    This function should return a list or set of words from the given filename.
    """
    #set(open("positive.txt").read().split('\n'))
    return set(open(filename).read().split('\n'))
    
    # YOUR CODE HERE

def updateFunction(newValues, runningCount):
    if runningCount is None:
        runningCount = 0
    return sum(newValues, runningCount)  # add the new values with the previous running count to get the new count

def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1])

    # Each element of tweets will be the text of a tweet.
    # You need to find the count of all the positive and negative words in these tweets.
    # Keep track of a running total counts and print this at every time step (use the pprint function).
    # YOUR CODE HERE
    
    #positivetwords = tweets.map(lambda x: ("positive", 1) if x in pwords )
    #negativewords = tweets.map(lambda x: ("negative", 1) if x in nwords )
    tweets = tweets.flatMap(lambda x: x.split(" "))
    tweets = tweets.map(lambda x: ("positive", 1) if x in pwords else ( ("negative", 1) if x in nwords else ("unknown", 1)))
    #print(tweets.count())
    tweets = tweets.filter(lambda x: x[0]!="unknown")
    #print(tweets.count())
    currentStateCount = tweets.reduceByKey(lambda a,b:a+b) 
    totalCount = currentStateCount.updateStateByKey(updateFunction)
    totalCount.pprint()
    
    # Let the counts variable hold the word counts for all time steps
    # You will need to use the foreachRDD function.
    # For our implementation, counts looked like:
    #   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
    counts = []
    currentStateCount.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    
    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)

    return counts


if __name__=="__main__":
    main()

