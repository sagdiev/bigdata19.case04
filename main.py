from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression

import config


DATA_CSV = config.BUILDDIR / 'bd_lab_small_sample.csv'
DATA_PARQUET = DATA_CSV.with_suffix('.parquet')

spark = SparkContext.getOrCreate()
sql = SQLContext(spark)


def explore():

    #datafile = spark.textFile(f'file://{DATA_CSV}')
    #breakpoint()
    #type(datafile)
    #datafile.first()

    df = sql.read.csv(str(DATA_CSV), header=True, inferSchema='true')
    #breakpoint()
    #df.show()
    #df.printSchema()

    df.write.parquet(str(DATA_PARQUET))
    #breakpoint()
    #df.groupby('event').count().show()
    #for c in df.columns: print(c); df.groupby(c).count().show()


def model():

    data = sql.read.parquet(str(DATA_PARQUET))
    data.createOrReplaceTempView('data')
    sample = sql.sql('''
        select
            hash_number_A
            ,interest_1
            ,phone_price_category
            ,sum(cost) as label
        from data
        group by hash_number_A, interest_1, phone_price_category''')
            # ,phone_price_category

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
