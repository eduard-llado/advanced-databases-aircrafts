import shutil

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.sql import SparkSession


def analysis(sc):
    """
    This pipeline trains a decision tree classifier for predicting if an aircraft is going to interrupt
    its operation unexpectedly in the next seven days.
    """

    sess = SparkSession(sc)

    """Create two datasets (training and validation) and format them."""

    # Load the data stored in LIBSVM format as a DataFrame.
    data = sess.read.format("libsvm").option("numFeatures", "5").load("./matrix")
    data.show(5)

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer = VectorIndexer().setInputCol("features")\
        .setOutputCol("indexedFeatures")\
        .setMaxCategories(4)\
        .fit(data)

    # Split the data into training and test sets (30% held out for testing)

    (trainingData, testData) = data.randomSplit([0.7, 0.3], seed=1234)

    """Create and store the validated model. Included evaluation metrics: accuracy and recall."""

    # Train a DecisionTree model.
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    # Train model. This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").show(5)

    # Select (prediction, true label) and compute accuracy and recall.
    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction",
                                                           metricName="accuracy")
    accuracy = accuracy_evaluator.evaluate(predictions)
    print("Accuracy = %g " % accuracy)
    recall_evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction",
                                                         metricName="weightedRecall")
    recall = recall_evaluator.evaluate(predictions)
    print("Recall (Weighted) = %g " % recall)

    # Model summary
    treeModel = model.stages[2]
    print(treeModel)

    dir_path = "./model"

    shutil.rmtree(dir_path, ignore_errors=True)

    model.save(dir_path)
