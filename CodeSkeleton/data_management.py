import pyspark
from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql import SparkSession
import operator
from pyspark.sql.functions import avg


DWusername = "eduard.llado"
DWpassword = "DB100200"

def management(sc):
    sess = SparkSession(sc)

    DW = (sess.read
        .format("jdbc")
        .option("driver","org.postgresql.Driver")
        .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require")
        .option("dbtable", "public.aircraftutilization")
        .option("user", DWusername)
        .option("password", DWpassword)
        .load())

    KPIs = (DW
        .select(["aircraftid", "flighthours", "flightcycles", "delayedminutes"])
        # change flighthours to float, flightcycles and delayedminutes to int
        .rdd)

    # sensors = (sc.wholeTextFiles("./resources/trainingData/*.csv")
    #     .cache())
    sensors = (sc.textFile("./resources/trainingData/*.csv")
        .cache())

    for x in sensors.collect():
        print(x)
