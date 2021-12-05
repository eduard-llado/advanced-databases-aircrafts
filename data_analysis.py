import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql.functions import col

def analysis(sc):
    sess = SparkSession(sc)

    data = (sess.read.option("header", True).csv("./matrix")
            .withColumn("SensorAVG", col("SensorAVG").cast(DoubleType()))
            .withColumn("FH", col("FH").cast(DoubleType()))
            .withColumn("FC", col("FC").cast(IntegerType()))
            .withColumn("DM", col("DM").cast(IntegerType())))

    labelIndexer = StringIndexer(inputCol="Label", outputCol="indexedLabel").fit(data)
    vectorAssembler = VectorAssembler(inputCols=["SensorAVG", "FH", "FC", "DM"], outputCol="features")
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features")
    pipeline = Pipeline(stages=[labelIndexer, vectorAssembler, dt])

    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    model = pipeline.fit(trainingData)
    predictions = model.transform(testData)
    predictions.select("prediction", "indexedLabel", "features").show(5)
    
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(accuracy)

