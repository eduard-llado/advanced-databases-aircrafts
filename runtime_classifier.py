import shutil

from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession

username = "yinlena.xu"
password = "DB040200"


def evaluation(sc, aircraft, date):
    """
    This pipeline predicts if an aircraft is going to go for unscheduled maintenance in
    the next seven days, given a record (<aircraft, date, FH, FC, DM, AVG(sensor)>).
    """

    """Given an aircraft and a date, replicates the data management pipeline and prepares 
    the tuple to be inputted into the model."""

    sess = SparkSession(sc)

    sensors = (sc.wholeTextFiles("./resources/trainingData/" + date + "*" + aircraft + ".csv")
               .map(lambda t: (aircraft, t[1]))
               .flatMapValues(lambda t: t.split("\n")[1:-1])
               .mapValues(lambda t: (t.split(" ")[0], t.split(";")[2]))
               .map(lambda t: ((t[0], str(t[1][0])), (float(t[1][1]), 1)))
               .reduceByKey(lambda t1, t2: (t1[0] + t2[0], t1[1] + t2[1]))
               .mapValues(lambda t: t[0] / t[1]))

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
            .filter(lambda t: t[0] == aircraft)
            .filter(lambda t: str(t[1]) == "20" + date[4:] + "-" + date[2:4] + "-" + date[:2])
            .map(lambda t: ((t[0], str(t[1])), (float(t[2]), int(t[3]), int(t[4])))))

    enrichedSensors = sensors.join(kpis).mapValues(lambda t: (t[1][0], t[1][1], t[1][2], t[0]))

    inputRecord = (enrichedSensors.map(lambda t: LabeledPoint(-1, [t[1][3], t[1][0], t[1][1], t[1][2]])))

    dir_path = "./input"
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))

    MLUtils.saveAsLibSVMFile(inputRecord, dir_path)

    """Classifies the record and outputs maintenance / no maintenance."""

    data = sess.read.format("libsvm").option("numFeatures", "5").load(dir_path)
    inputRecord = data.toDF("indexedLabel", "indexedFeatures")

    model = DecisionTreeClassificationModel.load("./model")
    predictions = model.transform(inputRecord)
    predictions.select("prediction", "indexedFeatures").show()
