from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

import data_management


def evaluation(sc, aircraft, date):
    """
    This pipeline predicts if an aircraft is going to go for unscheduled maintenance in the
    next seven days given <aircraft, date>, which corresponds to a record <AVG(sensor), FH, FC, DM>.
    """

    sess = SparkSession(sc)

    """Given an aircraft and a date, replicates the data management pipeline and prepares 
    the tuple to be inputted into the model."""

    data_management.management(sc, aircraft, date)

    data = sess.read.format("libsvm").option("numFeatures", "5").load("./matrix")
    data.show()

    """Classifies the record and outputs maintenance / no maintenance."""

    model = PipelineModel.load("./model")
    predictions = model.transform(data)
    predictions.select("prediction", "indexedFeatures").show()
