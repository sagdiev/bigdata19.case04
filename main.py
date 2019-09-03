from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.stat import Correlation, Summarizer
import sys

import config


DATA_CSV = config.BUILDDIR / 'bd_lab_small_sample.csv'
DATA_PARQUET = DATA_CSV.with_suffix('.parquet')

spark = SparkContext.getOrCreate()
sql = SQLContext(spark)


def convert():
    """Convert CSV to Parquet."""
    df = sql.read.csv(str(DATA_CSV), header=True, inferSchema='true')
    for field in ['cost', 'call_duration_minutes', 'data_volume_mb', 'LAT', 'LON']:
        df = df.withColumn(field, df[field].cast(FloatType()))
    df.write.parquet(str(DATA_PARQUET))


def explore():

    df = sql.read.parquet(str(DATA_PARQUET))
    df.printSchema()
    breakpoint()

    df.agg({'target': 'max'}).collect()

    from pyspark.sql.types import FloatType
    df = df.withColumn('cost', df['cost'].cast(FloatType()))

    df.select(df.columns[:10]).show()

    from math import ceil
    for i in range(ceil(len(df.columns) / 5)):
        df.select(df.columns[i*5:(i+1)*5]).show()

    df.groupby(df['phone_price_category']).count().toPandas().to_csv('build/test.csv')

    df.groupby(df['phone_price_category']).count().coalesce(1).write.csv('build/test2.csv', header=True)

    df = df.withColumn('phone_price_category', df['phone_price_category'].cast(FloatType()))
    df.corr('cost', 'phone_price_category')

    df.groupBy('hash_number_A')\
        .agg({'cost': 'sum', 'phone_price_category': 'max'})\
        .dropna().corr('sum(cost)', 'max(phone_price_category)')

    df.groupBy('hash_number_A')\
        .agg({'cost': 'sum', 'phone_price_category': 'max'})\
        .dropna()\
        .explain()

    df.crosstab('device_type', 'phone_price_category')

    df.fillna(0, ['phone_price_category']).crosstab('device_type', 'phone_price_category').show()

    df.cube('device_type', 'phone_price_category').sum().show()
    df.cube('device_type', 'phone_price_category').sum('cost', 'target').show()


def basic_statistics():
    """Basic statistics."""

    df = sql.read.parquet(str(DATA_PARQUET))

    numeric = ['cost', 'call_duration_minutes', 'data_volume_mb']
    assemble = VectorAssembler(inputCols=numeric, outputCol='features')
    features = assemble.transform(df.dropna(subset=numeric+['target']))

    breakpoint()

    # summarize
    summarize = Summarizer().metrics('mean', 'variance', 'count', 'numNonZeros', 'max', 'min', 'normL2', 'normL1')
    features.select(summarize.summary(features['features'])).show(truncate=False)

    # correlations
    r1 = Correlation.corr(features, 'features', 'pearson').head()[0]
    small = features.sample(fraction=0.1, seed=100500)
    r2 = Correlation.corr(small, 'features', 'spearman').head()[0]


def classify_target():
    """Forecast binary target."""

    df = sql.read.parquet(str(DATA_PARQUET))
    features = ['cost', 'call_duration_minutes', 'data_volume_mb']
    variables = features + ['test_flag', 'target']

    pipeline_prepare = Pipeline(stages=[
        VectorAssembler(inputCols=features, outputCol='features'),
        ])

    prepared = pipeline_prepare.fit(df).transform(df.dropna(subset=variables))
    training = prepared.filter(col('test_flag') == 0)
    testing = prepared.filter(col('test_flag') == 1)
    training_small = training.sample(fraction=0.3, seed=100500)

    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='target')

    breakpoint()

    # Logistic regression

    classifier = LogisticRegression(regParam=0.3, elasticNetParam=0,
        featuresCol='features', labelCol='target', predictionCol='prediction', probabilityCol='probability')
    model = classifier.fit(training_small)
    predicted = model.transform(testing)
    print('Test Area Under ROC: ', evaluator.evaluate(predicted))

    breakpoint()

    # Decision Tree Classifier

    classifier = DecisionTreeClassifier(featuresCol='features', labelCol='target', maxDepth=3)
    model = classifier.fit(training_small)
    predicted = model.transform(testing)
    print('Test Area Under ROC: ', evaluator.evaluate(predicted))

    breakpoint()

    # Random Forest Classifier
    rf = RandomForestClassifier(featuresCol='features', labelCol='label')
    model = classifier.fit(training_small)
    predicted = model.transform(testing)
    print('Test Area Under ROC: ', evaluator.evaluate(predicted))

    breakpoint()


def model():

    data = sql.read.parquet(str(DATA_PARQUET))
    data.createOrReplaceTempView('data')
    sample = sql.sql('''
        select
            hash_number_A
            ,interest_1
            ,interest_2
            ,interest_3
            ,interest_4
            ,interest_5
            ,device_type
            ,phone_price_category
            ,sum(cost) as label
        from data
        group by {", ".join(str(n) for n in range(1, 8+1))}''')
    breakpoint()

    pipeline = Pipeline(stages=[
        StringIndexer(inputCol='interest_1', outputCol='interest'),
        StringIndexer(inputCol='phone_price_category', outputCol='phone_price'),
        VectorAssembler(inputCols=['interest', 'phone_price'], outputCol='features'),
        ])
    model_data = pipeline.fit(sample)

    sample = model_data.transform(sample)

    # 'gaussian', 'binomial', 'poisson', 'gamma', 'tweedie'

    regression = GeneralizedLinearRegression(family='gaussian', labelCol='label', featuresCol='features', maxIter=10, regParam=0.3)
    model = regression.fit(sample)
    breakpoint()


def main():
    print('Executing main()')
    exec(sys.argv[1])


if __name__ == '__main__':
    main()
