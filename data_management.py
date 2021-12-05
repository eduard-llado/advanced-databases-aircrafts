from pyspark.sql import Row
from pyspark.sql import SparkSession
from datetime import timedelta

username = "eduard.llado"
password = "DB100200"


def management(sc):
    """
    This pipeline generates a matrix where the rows denote the information of an aircraft per day,
    and the columns refer to the FH, FC and DM KPIs, and the average measurement of the 3453 sensor.
    """

    sess = SparkSession(sc)

    """Read the sensor measurements (extracted from the CSV files) for each aircraft and average it per day."""

    sensors = (sc.wholeTextFiles("./resources/trainingData/*.csv")
               .map(lambda t: (t[0].split("/")[-1][-10:-4], t[1]))          # (1)
               .flatMapValues(lambda t: t.split("\n")[1:-1])                # (2)
               .mapValues(lambda t: (t.split(" ")[0], t.split(";")[2]))     # (3)
               .map(lambda t: ((t[0], str(t[1][0])), (float(t[1][1]), 1)))  # (4)
               .reduceByKey(lambda t1, t2: (t1[0] + t2[0], t1[1] + t2[1]))  # (5)
               .mapValues(lambda t: t[0] / t[1]))                           # (6)

    # (1) Get aircraft ID from file path and set it as key
    # (2) Remove special rows (column header and final empty row) and split events
    # (3) Keep only day and sensor value
    # (4) Move day to key and prepare to get average
    # (5) Compute sensor sum and count
    # (6) Compute sensor average
    # sensors output: ((aircraft, date), AVG(sensor)

    """Enrich the sensor average with the KPIs from the Data Warehouse at the same granularity level."""

    DW = (sess.read
          .format("jdbc")
          .option("driver", "org.postgresql.Driver")
          .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require")
          .option("dbtable", "public.aircraftutilization")
          .option("user", username)
          .option("password", password)
          .load())

    kpis = (DW.select("aircraftid", "timeid", "flighthours", "flightcycles", "delayedminutes")
            .rdd
            .map(lambda t: ((t[0], str(t[1])), (float(t[2]), int(t[3]), int(t[4])))))

    # kpis output: ((aircraft, date), (FH, FC, DM))

    enrichedSensors = sensors.join(kpis).mapValues(lambda t: (t[1][0], t[1][1], t[1][2], t[0]))

    # enrichedSensors output: ((aircraft, date), (FH, FC, DM, AVG(sensor)))

    """Add maintenance label for supervised algorithm."""

    AMOS = (sess.read
            .format("jdbc")
            .option("driver", "org.postgresql.Driver")
            .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require")
            .option("dbtable", "oldinstance.maintenanceevents")
            .option("user", username)
            .option("password", password)
            .load())

    labels = (AMOS.select("aircraftregistration", "starttime", "subsystem", "kind")
              .rdd
              .filter(lambda t: t[2] == "3453")                                             # (1)
              .filter(lambda t: t[3] in ("Delay", "Safety", "AircraftOnGround"))            # (2)
              .flatMapValues(lambda t: (str((t.date() - timedelta(i))) for i in range(7)))  # (3)
              .map(lambda t: ((t[0], t[1]), 1))                                             # (4)
              .distinct())

    # (1) Filter aircraft navigation subsystem (ATA code: 3453)
    # (2) Get unscheduled events
    # (3) Reformat date
    # (4) Set aircraft and date as key
    # labels output: ((aircraft, date), 1) for all unscheduled events

    labeledSensors = enrichedSensors\
        .leftOuterJoin(labels)\
        .mapValues(lambda t: t[0] + ({1: "Maintenance", None: "NonMaintenance"}[t[1]],))

    # labeledSensors output: ((aircraft, date), (FH, FC, DM, AVG(sensor), label))

    """Generate a matrix with the gathered data and store it."""

    sess.createDataFrame(
        labeledSensors.map(lambda t: Row(FH=t[1][0], FC=t[1][1], DM=t[1][2], SensorAVG=t[1][3], Label=t[1][4])))\
        .write.mode("overwrite")\
        .option("header", True)\
        .csv("matrix")
