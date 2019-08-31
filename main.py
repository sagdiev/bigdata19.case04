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

    #df = sql.read.csv(str(DATA_CSV), header=True, inferSchema='true')
    #breakpoint()
    #df.show()
    #df.printSchema()

    #df.write.parquet(str(DATA_PARQUET))
    #breakpoint()
    #df.groupby('event').count().show()
    #for c in df.columns: print(c); df.groupby(c).count().show()

    df = sql.read.parquet(str(DATA_PARQUET))
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
