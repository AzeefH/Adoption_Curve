###################### SSH ######################

#Change Master Public DNS Name

url=hadoop@ip-172-22-143-129.ec2.internal
ssh -i ~/emr_key.pem $url

pkg_list=com.databricks:spark-avro_2.11:4.0.0,org.apache.hadoop:hadoop-aws:2.7.1
pyspark --packages $pkg_list --num-executors 25 --conf "spark.executor.memoryOverhead=2048" --executor-memory 9g --conf "spark.driver.memoryOverhead=6144" --driver-memory 50g --executor-cores 3 --driver-cores 5 --conf "spark.default.parallelism=150" --conf "spark.sql.shuffle.partitions=150" --conf "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version=2"

####
#pkg_list=com.databricks:spark-avro_2.11:4.0.0,org.apache.hadoop:hadoop-aws:2.7.1
#pyspark --packages $pkg_list


from pyspark import SparkContext, SparkConf, HiveContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.sql.functions as F
import pyspark.sql.types as T
import csv
import pandas as pd
import numpy as np
import sys
from pyspark.sql import Window
from pyspark.sql.functions import rank, col
#import geohash2 as geohash
#import pygeohash as pgh
from functools import reduce
from pyspark.sql import *
from pyspark import StorageLevel

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)


############
#Parameters#
############
COUNTRY = 'MY'
YEAR = '202{0*,101,102,103,104,105,106,107,108}'


##################
#Device Behaviour#
##################
path = 's3a://ada-dev/azeef/lookup_tables/android_device_table_v202108.csv'
android_ref = spark.read.csv(path, header = True)

path = 's3a://ada-prod-data/etl/data/brq/agg/agg_brq/monthly/'+COUNTRY+'/'+YEAR+'/*.parquet'
android = spark.read.parquet(path).select('ifa', 'device', explode('app')).select('ifa', 'device.*', 'col.first_seen').filter((lower(col('platform')).like('android')))

android_join = android.join(android_ref, android.device_model == android_ref.Model ,how = 'left_outer').select('ifa','first_seen','Retail Branding','Marketing Name')\
            .fillna('Not Detected').withColumnRenamed('Marketing Name', 'Name')

#Filter only Samsung
android_join = android_join.filter(col('Retail Branding') == 'Samsung')

android_join.printSchema()

android_join = android_join.cache()
android_join.show(5,0)


# Add column date from first seen. Derice col year and month from date
android_df = android_join.withColumn('date', to_date('first_seen'))
android_df = android_df.withColumn('year', year('date'))
android_df = android_df.withColumn('month', month('date'))
android_df = android_df.cache()
android_df.show(5,0)


# Launch Tables, add columns of release month +1 and +2
path = 's3a://ada-dev/azeef/lookup_tables/samsung_launch.csv'
samsung_launch = spark.read.csv(path, header=True).withColumn('launch', to_date('Release_Date'))
#samsung_launch = samsung_launch.withColumn('Release_Month_plus1', col('Release_Month') + lit(1))
#samsung_launch = samsung_launch.withColumn('Release_Month_plus2', col('Release_Month_plus1') + lit(1)).withColumnRenamed('Name','Name_Temp')
samsung_launch = samsung_launch.withColumnRenamed('Name','Name_Temp')
samsung_launch = samsung_launch.drop(*['Release_Year','Release_Month','Release_Date'])
samsung_launch = samsung_launch.cache()
samsung_launch.show(20,0)

# Join with launch date csv that match Name, groupby ifa, sort date ascending, rank = 1
df1 = android_df.join(samsung_launch, (android_df.Name == samsung_launch.Name_Temp), how='inner')
window_spec = Window.partitionBy('ifa').orderBy(F.col("date").asc())
df2 = df1.withColumn('rank', F.row_number().over(window_spec))
df2 = df2.filter(F.col('rank')==1)
df2 = df2.drop(*['rank','year','month'])
df2 = df2.cache()
df2.show(5,0)


# Delta (date - Release_Date)
#df3 = df2.withColumn('Delta', (col('date') - col('launch'))) # delta in month & days
df3 = df2.withColumn('delta', datediff('date','launch')) # delta in days
df3 = df3.cache()
df3.show(5,0)
df3.printSchema()

df4 = df3.filter(~col('delta').like('%-%'))
df4 = df4.cache()
df4.show(5,0)

# Groupby delta and count ifas to get delta thresholds
#df5 = df4.groupBy('delta').agg(count('ifa').alias('count')).sort(col('delta'), ascending=True)
#df5.show(100,0)

#path = 's3a://ada-dev/azeef/projects/'+COUNTRY+'/202109/device_launch/samsung_delta_new'
#df5.coalesce(1).write.csv(path, header=True)

'''
Segments	Threshold (days)
Innovators	<7
Early Adopters	7-108
Early Majority	109-194
Late Majority	195-341
Laggards	>341
'''

##########################################
# Extarct IFAs of each adoption segment #
##########################################
# Define each segment

## Innovators
innovator = df4.filter(col('delta') < 7)
innovator = innovator.withColumn('quarter', F.lit('202109'))\
                .withColumn('adoption_type', F.lit('samsung_high_end_phones'))\
                .withColumn('segment_id', F.lit('SEG_TEL_NS_00008'))\
                .withColumn('segment', F.lit('Innovator'))\
                .withColumn('device_brand', F.lit('Samsung'))\
                .withColumnRenamed('Name_Temp', 'device_name')\
                .withColumnRenamed('launch', 'launch_date')
innovator = innovator.select('quarter','adoption_type','ifa','segment_id','segment','device_brand','device_name','launch_date','first_seen','delta')
innovator = innovator.cache()
innovator.select('ifa').distinct().count() # 7403
innovator.show(5,0)
innovator.printSchema()

+-------+-----------------------+------------------------------------+----------------+---------+------------+-------------------+-----------+-------------------+-----+
|quarter|adoption_type          |ifa                                 |segment_id      |segment  |device_brand|device_name        |launch_date|first_seen         |delta|
+-------+-----------------------+------------------------------------+----------------+---------+------------+-------------------+-----------+-------------------+-----+
|202109 |samsung_high_end_phones|05a0b1c5-2f5f-4783-a2c2-42960a190b2a|SEG_TEL_NS_00008|Innovator|Samsung     |Galaxy S20         |2020-03-06 |2020-03-07 04:09:48|1    |
|202109 |samsung_high_end_phones|2b6e7a52-5f06-439d-96a9-6fcd5b00896e|SEG_TEL_NS_00008|Innovator|Samsung     |Galaxy S20         |2020-03-06 |2020-03-07 09:43:06|1    |
|202109 |samsung_high_end_phones|2bd42cd7-93b8-4217-9e96-9a03430e203a|SEG_TEL_NS_00008|Innovator|Samsung     |Galaxy S21 Ultra 5G|2021-01-29 |2021-01-29 16:33:33|0    |
|202109 |samsung_high_end_phones|2d7ef709-87ae-4430-b1e8-45de09628581|SEG_TEL_NS_00008|Innovator|Samsung     |Galaxy S20 Ultra 5G|2020-03-06 |2020-03-07 19:00:22|1    |
|202109 |samsung_high_end_phones|37cd2e91-9307-45ba-a47e-244df1fe0d9d|SEG_TEL_NS_00008|Innovator|Samsung     |Galaxy S21+ 5G     |2021-01-29 |2021-02-01 10:05:09|3    |
+-------+-----------------------+------------------------------------+----------------+---------+------------+-------------------+-----------+-------------------+-----+

root
 |-- quarter: string (nullable = false)
 |-- adoption_type: string (nullable = false)
 |-- ifa: string (nullable = false)
 |-- segment_id: string (nullable = false)
 |-- segment: string (nullable = false)
 |-- device_brand: string (nullable = false)
 |-- device_name: string (nullable = true)
 |-- launch_date: date (nullable = true)
 |-- first_seen: timestamp (nullable = true)
 |-- delta: integer (nullable = true)

## Early Adopters
adopter = df4.filter(col('delta').between(7,108))
adopter = adopter.withColumn('quarter', F.lit('202109'))\
                .withColumn('adoption_type', F.lit('samsung_high_end_phones'))\
                .withColumn('segment_id', F.lit('SEG_TEL_NS_00009'))\
                .withColumn('segment', F.lit('Early_Adopter'))\
                .withColumn('device_brand', F.lit('Samsung'))\
                .withColumnRenamed('Name_Temp', 'device_name')\
                .withColumnRenamed('launch', 'launch_date')
adopter = adopter.select('quarter','adoption_type','ifa','segment_id','segment','device_brand','device_name','launch_date','first_seen','delta')
adopter = adopter.cache()
adopter.select('ifa').distinct().count() # 39087

## Early Majority
majority1 = df3.filter(col('delta').between(109,194))
majority1 = majority1.withColumn('quarter', F.lit('202109'))\
                .withColumn('adoption_type', F.lit('samsung_high_end_phones'))\
                .withColumn('segment_id', F.lit('SEG_TEL_NS_00010'))\
                .withColumn('segment', F.lit('Early_Majority'))\
                .withColumn('device_brand', F.lit('Samsung'))\
                .withColumnRenamed('Name_Temp', 'device_name')\
                .withColumnRenamed('launch', 'launch_date')
majority1 = majority1.select('quarter','adoption_type','ifa','segment_id','segment','device_brand','device_name','launch_date','first_seen','delta')
majority1 = majority1.cache()
majority1.select('ifa').distinct().count() # 98752

## Late Majority
majority2 = df3.filter(col('delta').between(195,341))
majority2 = majority2.withColumn('quarter', F.lit('202109'))\
                .withColumn('adoption_type', F.lit('samsung_high_end_phones'))\
                .withColumn('segment_id', F.lit('SEG_TEL_NS_00011'))\
                .withColumn('segment', F.lit('Late_Majority'))\
                .withColumn('device_brand', F.lit('Samsung'))\
                .withColumnRenamed('Name_Temp', 'device_name')\
                .withColumnRenamed('launch', 'launch_date')
majority2 = majority2.select('quarter','adoption_type','ifa','segment_id','segment','device_brand','device_name','launch_date','first_seen','delta')
majority2 = majority2.cache()
majority2.select('ifa').distinct().count() # 100298

## Laggards
laggard = df3.filter(col('delta') > 341)
laggard = laggard.withColumn('quarter', F.lit('202109'))\
                .withColumn('adoption_type', F.lit('samsung_high_end_phones'))\
                .withColumn('segment_id', F.lit('SEG_TEL_NS_00012'))\
                .withColumn('segment', F.lit('Laggard'))\
                .withColumn('device_brand', F.lit('Samsung'))\
                .withColumnRenamed('Name_Temp', 'device_name')\
                .withColumnRenamed('launch', 'launch_date')
laggard = laggard.select('quarter','adoption_type','ifa','segment_id','segment','device_brand','device_name','launch_date','first_seen','delta')
laggard = laggard.cache()
laggard.select('ifa').distinct().count() # 44872


## Join all segments
df = innovator.union(adopter).union(majority1).union(majority2).union(laggard)
#df.select('ifa').count() # 290412
#df.select('ifa').distinct().count() # 290412
df = df.cache()
df.show(20,0)
df.printSchema()


# Write out
path = 's3a://ada-dev/DA_datamart/project_segments/adoption/samsung_high_end_phones/MY/202109'
df.write.parquet(path)



############
