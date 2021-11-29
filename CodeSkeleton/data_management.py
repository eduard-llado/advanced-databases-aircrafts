import pyspark
from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql import SparkSession

username = "eduard.llado"
password = "DB100200"

def process(sc):
    sess = SparkSession(sc)

    DW = (sess.read
       .format("jdbc")
       .option("driver","org.postgresql.Driver")
       .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require")
       .option("dbtable", "public.aircraftdimension")
       .option("user", username)
       .option("password", password)
       .load())

    input = (sc.textFile("./resources/trainingData/311014-CHC-FUE-9429-XY-SHM.csv")
        .cache())
