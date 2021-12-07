import pyspark
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import PipelineModel
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark import SparkContext
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('WinePredict').getOrCreate()
dataset = spark.read.csv('winequality-white.csv',header='true', inferSchema='true', sep=';')

#Turn off warnings
spark.sparkContext.setLogLevel("OFF")

#Gets all features except for quality from dataset
featureColumns = [c for c in dataset.columns if c != 'quality']

#Assembler creates a vector of the features of each wine
assembler = VectorAssembler(inputCols=featureColumns, 
                            outputCol="features")

# Adds vector to data
dataDF = assembler.transform(dataset)
#Split assembled data
train,test = dataDF.randomSplit([0.8, 0.2])

#Creates linear regression model with feature column being feature and label column being quality
lr = LinearRegression(featuresCol="features", labelCol="quality")
#Fits the linear regression model to the training data
lrModel = lr.fit(train)
#Makes the prediction for the testing data
predictionsDF = lrModel.transform(test)

#Creates an evaluator using label column as quality and prediction column as prediction utilizing the rmse metric
evaluator = RegressionEvaluator(
    labelCol='quality', predictionCol="prediction", metricName="rmse")
#Creates rmse variable that evalutates the linear regression testing data
rmse = evaluator.evaluate(predictionsDF)
print("Linear Regression Root Mean Squared Error (RMSE) without hyperparameters = %g" % rmse)

#Splits initial training data before creating the feature columns
(trainingDF, testDF) = dataset.randomSplit([0.8, 0.2])
#Creates a pipeline since we need it for hyper parameters
pipeline = Pipeline(stages=[assembler, lr])
lrPipelineModel = pipeline.fit(trainingDF)

#Builds a grid for the 2 parameters of linear regressions and tests the different parameters to achieve the best results
#Tried different parameters but I think this one covered the best spread while not taking too long
search_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.0, 0.2, 0.4, 0.6, 0.8]) \
    .addGrid(lr.elasticNetParam, [0.2, 0.4, 0.6, 0.8]).build()

#Creates a cross validator utilizing the pipeline of linear regression model
cv = CrossValidator(estimator = pipeline, estimatorParamMaps = search_grid, evaluator = evaluator, numFolds = 3)
cvModel = cv.fit(trainingDF)
cvPrediction = cvModel.transform(testDF)
print("Linear Regression Root Mean Squared Error (RMSE) with hyperparameters = %g" % evaluator.evaluate(cvPrediction))

#Creates a model for the Random forest regression
#Tried 50/64/10 and got 33% accuracy
#Tried 200/256/20 and got slightly better results by less than .01 but takes a long time
rf = RandomForestRegressor(featuresCol="features", labelCol="quality", numTrees=100, maxBins=128, maxDepth=20, \
                           minInstancesPerNode=5, seed=33)
#Assembled the pipeline which is necessary for the model
rfPipeline = Pipeline(stages=[assembler, rf])

# train the random forest model on training data
rfPipelineModel = rfPipeline.fit(trainingDF)

rfTraining = rfPipelineModel.transform(trainingDF)
rfPredictions = rfPipelineModel.transform(testDF)
print("Random Forest Root Mean Squared Error (RMSE)  = %g" % evaluator.evaluate(rfPredictions))
