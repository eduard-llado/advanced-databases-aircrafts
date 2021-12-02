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

    sensors = (sc.wholeTextFiles("./resources/trainingData/*.csv")
    .map(lambda t: (t[0].split("/")[-1][-10:-4], t[1]))
    # ('XY-SHM', 'date;series;value\n2014-10-31 13:57:59.396;eventData;65.53802006168051\n2014-10-31 14:02:59.396;eventData;65.96371019435036\n2014-10-31 14:07:59.396;eventData;67.25221768056858\n2014-10-31 14:12:59.396;eventData;67.12692026132314\n2014-10-31 14:17:59.396;eventData;61.28435253159288\n2014-10-31 14:22:59.396;eventData;62.899959942434954\n')
    .flatMapValues(lambda t: t.split("\n")[1:-1])
    # ('XY-SHM', '2014-10-31 14:22:59.396;eventData;62.899959942434954')
    # ('XY-SHM', '2014-10-31 14:17:59.396;eventData;61.28435253159288')
    .mapValues(lambda t: (t.split(" ")[0], t.split(";")[2]))
    # ('XY-SHM', ('2014-10-31', '61.28435253159288'))
    # ('XY-SHM', ('2014-10-31', '62.899959942434954'))
    .map(lambda t: ((t[0], str(t[1][0])), (float(t[1][1]), 1)))
    # (('XY-SHM', '2014-10-31'), (61.28435253159288, 1))
    # (('XY-SHM', '2014-10-31'), (62.899959942434954, 1))
    .reduceByKey(lambda t1,t2: (t1[0]+t2[0], t1[1]+t2[1]))
    # (('XY-SHM', '2014-10-31'), (390.0651806719505, 6))
    .mapValues(lambda t: t[0]/t[1])
    # (('XY-SHM', '2014-10-31'), 65.01086344532509)
    )


    print(sensors.count())

    # DW = (sess.read
    #     .format("jdbc")
    #     .option("driver","org.postgresql.Driver")
    #     .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require")
    #     .option("dbtable", "public.aircraftutilization")
    #     .option("user", DWusername)
    #     .option("password", DWpassword)
    #     .load())
    #
    # KPIs = (DW
    #     .select("aircraftid", "timeid", "flighthours", "flightcycles", "delayedminutes")
    #     # Row(aircraftid='XY-POE', timeid=datetime.date(2012, 9, 18), flighthours=Decimal('1.416666666666670000'), flightcycles=Decimal('1.000000000000000000'), delayedminutes=Decimal('35.000000000000000000'))
    #     .rdd
    #     .map(lambda t: ((t[0], str(t[1])), (float(t[2]),int(t[3]),int(t[4]))))
    #     # (('XY-POE', '2012-09-18'), (1.41666666666667, 1, 35))
    #     )
