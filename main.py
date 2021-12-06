import os
import sys

import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession

import data_analysis
import data_management
import runtime_classifier

HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3"
PYSPARK_DRIVER_PYTHON = "python3"

if __name__ == "__main__":
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + "\\bin")
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

    conf = SparkConf()  # create the configuration
    conf.set("spark.jars", JDBC_JAR)

    spark = SparkSession.builder \
        .config(conf=conf) \
        .master("local") \
        .appName("Training") \
        .getOrCreate()

    sc = pyspark.SparkContext.getOrCreate()

    # Create and point to your pipelines here
    if len(sys.argv) < 2:
        print("Wrong number of parameters, usage: (management, analysis, <aircraft date>)")
        exit()
    if sys.argv[1] == "management":
        data_management.management(sc)
    elif sys.argv[1] == "analysis":
        data_analysis.analysis(sc)
    else:
        aircraft = str(sys.argv[1])
        date = str(sys.argv[2])
        runtime_classifier.evaluation(sc, aircraft, date)
