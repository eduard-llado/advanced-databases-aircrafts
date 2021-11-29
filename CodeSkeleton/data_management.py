import pyspark
from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql import SparkSession

username = "nom.cognom"
password = "DBddnnyy"

def process(sc):
    sess = SparkSession(sc)

    AMOS = (sess.read
        .format("jdbc")
        .option("driver","org.postgresql.Driver")
        .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require")
        .option("dbtable", "oldinstance.MaintenanceEvents")
        .option("user", username)
        .option("password", password)
        .load())

    AIMS = (sess.read
        .format("jdbc")
        .option("driver","org.postgresql.Driver")
        .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/AIMS?sslmode=require")
        .option("dbtable", "public.flights")
        .option("user", username)
        .option("password", password)
        .load())

    input = (sc.textFile("./resources/trainingData/311014-CHC-FUE-9429-XY-SHM.csv.csv")
        .cache())
