import pyspark
from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql import SparkSession
import operator
from pyspark.sql.functions import avg


username = "eduard.llado"
password = "DB100200"

def management(sc):
    sess = SparkSession(sc)

    ### Read the sensor measurements (extracted from the CSV files) for each aircraft and average it per day ###
    
    sensors = (sc.wholeTextFiles("./resources/trainingData/*.csv")          #                                                                             (1)
               .map(lambda t: ((t[0].split("/")[-1][-10:-4], t[1])))        # Get aircraft ID from file path and set it as key                            (2)
               .flatMapValues(lambda t: t.split("\n")[1:-1])                # Remove special rows (column names and final empty row) and split events     (3)
               .mapValues(lambda t: (t.split(" ")[0], t.split(";")[2]))     # Keep only day and sensor value                                              (4)
               .map(lambda t: ((t[0], str(t[1][0])), (float(t[1][1]), 1)))  # Move day to key and prepare to get average                                  (5)
               .reduceByKey(lambda t1, t2: (t1[0] + t2[0], t1[1] + t2[1]))  # Get sensor sum and count                                                    (6)
               .mapValues(lambda t: t[0] / t[1]))                           # Get sensor average                                                          (7)

    # (1) ('file:/path/to/290716-YSB-TNA-7922-XY-SHM.csv', 'date;series;value\n2014-10-31 13:57:59.396;eventData;65.53802006168051\n ... \n')
    # (2) ('XY-SHM', 'date;series;value\n2014-10-31 13:57:59.396;eventData;65.53802006168051\n ... \n')
    # (3) ('XY-SHM', '2014-10-31 14:22:59.396;eventData;62.899959942434954')
    # (4) ('XY-SHM', ('2014-10-31', '61.28435253159288'))
    # (5) (('XY-SHM', '2014-10-31'), (61.28435253159288, 1))
    # (6) (('XY-SHM', '2014-10-31'), (390.0651806719505, 6))
    # (7) (('XY-SHM', '2014-10-31'), 65.01086344532509)

    ### Enrich the sensor average with the KPIs from the Data Warehouse at the same granularity level ###

    DW = (sess
          .read
          .format("jdbc")
          .option("driver","org.postgresql.Driver")
          .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require")
          .option("dbtable", "public.aircraftutilization")
          .option("user", username)
          .option("password", password)
          .load())

    kpis = (DW
            .select("aircraftid", "timeid", "flighthours", "flightcycles", "delayedminutes")
            .rdd
            .map(lambda t: ((t[0], str(t[1])), (float(t[2]), int(t[3]), int(t[4])))))
    
    # kpis: ((aircraft-id, date), (FH, FC, DM)

    enrichedSensors = sensors.join(kpis).map(lambda t: (t[0], (t[1][1][0], t[1][1][1], t[1][1][2], t[1][0])))

    # enrichedSensors: ((aircraft-id, date), (FH, FC, DM, AVG(sensor)))

    ### Add maintenance label for supervised algorithm ###

    AMOS = (sess.read
        .format("jdbc")
        .option("driver","org.postgresql.Driver")
        .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require")
        .option("dbtable", "oldinstance.maintenanceevents")
        .option("user", username)
        .option("password", password)
        .load())

    labels = (AMOS
              .select("aircraftregistration")
              .rdd)


    # - Generate a matrix with the gathered data and store it.
