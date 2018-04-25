import org.apache.spark.sql.SparkSession
//import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

//import org.apache.spark.ml.classification.NaiveBayes
//import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.feature.{HashingTF, IDF}

object spam {

  def main(args: Array[String]): Unit = {

    // start spark session
    val spark = SparkSession
      .builder
      .master("local")
      .appName("Classification")
      .getOrCreate()

    // load data as spark-datasets
    val spam_training = spark.read.textFile("src/main/resources/spam_training.txt")
    val spam_testing = spark.read.textFile("src/main/resources/spam_testing.txt")
    val nospam_training = spark.read.textFile("src/main/resources/nospam_training.txt")
    val nospam_testing = spark.read.textFile("src/main/resources/nospam_testing.txt")

    // implement: convert datasets to either rdds or dataframes (your choice) and build your pipeline
    val spam =  spam_training.rdd
    val ham = nospam_training.rdd

    val spamtest =  spam_testing.rdd
    val hamtest = nospam_testing.rdd

    val tf = new HashingTF(numFeatures = 100)
    val spamFeatures = spam.map(email => tf.transform(email.split(" ")))
    val hamFeatures = ham.map(email => tf.transform(email.split(" ")))
    val spamFeaturest = spamtest.map(email => tf.transform(email.split(" ")))
    val hamFeaturest = hamtest.map(email => tf.transform(email.split(" ")))

    val idf  = new IDF().fit(spamFeatures++hamFeatures++spamFeaturest++hamFeaturest)
    val spamFeaturesidf =idf.transform(hamFeatures)
    val hamFeaturesidf =idf.transform(hamFeatures)
    val spamFeaturestidf =idf.transform(spamFeaturest)
    val hamFeaturestidf =idf.transform(hamFeaturest)

    val positiveExamples = spamFeaturesidf.map(features => LabeledPoint(1, features))
    val negativeExamples = hamFeaturesidf.map(features => LabeledPoint(0, features))
    val trainingData = positiveExamples ++ negativeExamples
    trainingData.cache() // Cache data since Logistic Regression is an iterative algorithm.


    val positiveExamplest = spamFeaturestidf.map(features => LabeledPoint(1, features))
    val negativeExamplest = hamFeaturestidf.map(features => LabeledPoint(0, features))
    val testData = positiveExamplest ++ negativeExamplest
    testData.cache() // Cache data since Logistic Regression is an iterative algorithm.


    val model = NaiveBayes.train(trainingData, lambda = 1.0, modelType = "multinomial")

    val predictionAndLabel = testData.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testData.count()
    println("Test set accuracy = " + accuracy)


    spark.stop()
  }
}
