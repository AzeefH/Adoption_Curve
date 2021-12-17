###################### SSH ######################

#Change Master Public DNS Name

url=hadoop@ip-172-22-143-234.ec2.internal
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



#############
#Device REF#
#############
path = 's3a://ada-dev/azeef/lookup_tables/android_device_table_v202108.csv'
android_ref = spark.read.csv(path, header = True)


################################
# Loop to Optimise performance #
################################
COUNTRY = 'ID'
list = ['202001','202002','202003','202004','202005','202006','202007','202008','202009','202010','202011','202012','202101','202102','202103','202104','202105','202106','202107','202108']



for i in list:
    print(i)
    path = 's3a://ada-prod-data/etl/data/brq/agg/agg_brq/monthly/'+COUNTRY+'/'+i+'/*.parquet'
    android = spark.read.parquet(path).select('ifa', 'device', explode('app')).select('ifa', 'device.*', 'col.first_seen').filter((lower(col('platform')).like('android')))
    android_join = android.join(android_ref, android.device_model == android_ref.Model ,how = 'left_outer').select('ifa','first_seen','Retail Branding','Marketing Name').fillna('Not Detected').withColumnRenamed('Marketing Name', 'Name')
    #### Filter only Samsung
    android_join = android_join.filter(col('Retail Branding') == 'Samsung')
    #android_join.printSchema()
    android_join = android_join.cache()
    android_join.show(5,0)
    #### Add column date from first seen. Derice col year and month from date
    android_df = android_join.withColumn('date', to_date('first_seen'))
    android_df = android_df.withColumn('year', year('date'))
    android_df = android_df.withColumn('month', month('date'))
    android_df = android_df.cache()
    android_df.show(5,0)
    #### Launch Tables, add column launch
    path = 's3a://ada-dev/azeef/lookup_tables/samsung_launch_id.csv'
    samsung_launch = spark.read.csv(path, header=True).withColumn('launch', to_date('Release_Date'))
    samsung_launch = samsung_launch.withColumnRenamed('Name','Name_Temp')
    samsung_launch = samsung_launch.drop(*['Release_Year','Release_Month','Release_Date'])
    samsung_launch = samsung_launch.cache()
    samsung_launch.show(20,0)
    #### Join with launch date csv that match Name, groupby ifa, sort date ascending, rank = 1
    df1 = android_df.join(samsung_launch, (android_df.Name == samsung_launch.Name_Temp), how='inner')
    window_spec = Window.partitionBy('ifa').orderBy(F.col("date").asc())
    df2 = df1.withColumn('rank', F.row_number().over(window_spec))
    df2 = df2.filter(F.col('rank')==1)
    df2 = df2.drop(*['rank','year','month','Retail Branding'])
    df2 = df2.cache()
    df2.show(5,0)
    #### Delta (date - Release_Date)
    df3 = df2.withColumn('delta', datediff('date','launch')) # delta in days
    df3 = df3.cache()
    df3.show(5,0)
    #df3.printSchema()
    #### Remove negative deltas
    df4 = df3.filter(~col('delta').like('%-%'))
    df4 = df4.cache()
    df4.show(5,0)
    # Save df4 in parquet without the grouping to consrve columns
    #save_path = 's3a://ada-dev/azeef/projects/'+COUNTRY+'/202109/device_launch/high_end_samsung/parquet/'+i+''
    #df4.write.parquet(save_path, , mode='overwrite')
    #### Groupby delta and count ifas to get delta thresholds
    df5 = df4.groupBy('delta').agg(countDistinct('ifa').alias('count')).sort(col('delta'), ascending=True)
    df5.show(10,0)
    #### Write parquet for each month
    save_path = 's3a://ada-dev/azeef/projects/'+COUNTRY+'/202109/device_launch/high_end_samsung_delta_new/parquet/'+i+''
    df5.write.parquet(save_path, mode='overwrite')



##############################
# Combine csv for all months #
##############################
path = 's3a://ada-dev/azeef/projects/'+COUNTRY+'/202109/device_launch/high_end_samsung_delta_new/parquet/202*/*.parquet'
data = spark.read.parquet(path)

# GroupBy delta agg SUM count
data_df = data.groupBy('delta').agg(sum('count').alias('sum_count')).sort(col('delta'), ascending=True)
data_df = data_df.cache()
data_df.show(5,0)

# Write combined csv of all months
save_path = 's3a://ada-dev/azeef/projects/'+COUNTRY+'/202109/device_launch/high_end_samsung_delta_new/csv/202001_202108'
data_df.coalesce(1).write.csv(save_path, header=True, mode='overwrite')



# Analyse the csv and get the threshold deltas
# Example:
'''
Segments	Threshold (days)
Innovators	<23
Early Adopters	23-128
Early Majority	129-332
Late Majority	333-469
Laggards	>469
'''


##########################################
# Extarct IFAs of each adoption segment #
##########################################
# Load data
path = 's3a://ada-dev/azeef/projects/'+COUNTRY+'/202109/device_launch/high_end_samsung/parquet/*/*.parquet'
df4 = spark.read.parquet(path)

# Define each segment
## Innovators
innovator = df4.filter(col('delta') < 23)
innovator = innovator.withColumn('quarter', F.lit('202109'))\
                .withColumn('adoption_type', F.lit('samsung_high_end_phones'))\
                .withColumn('segment_id', F.lit('SEG_TEL_NS_00013'))\
                .withColumn('segment', F.lit('Innovator'))\
                .withColumn('device_brand', F.lit('Samsung'))\
                .withColumnRenamed('Name_Temp', 'device_name')\
                .withColumnRenamed('launch', 'launch_date')
innovator = innovator.select('quarter','adoption_type','ifa','segment_id','segment','device_brand','device_name','launch_date','first_seen','delta')
innovator = innovator.cache()
innovator.select('ifa').distinct().count() # 30740
innovator.select('ifa').count() # 30741
innovator.show(5,0)
innovator.printSchema()

## Early Adopters
adopter = df4.filter(col('delta').between(22,129))
adopter = adopter.withColumn('quarter', F.lit('202109'))\
                .withColumn('adoption_type', F.lit('samsung_high_end_phones'))\
                .withColumn('segment_id', F.lit('SEG_TEL_NS_00014'))\
                .withColumn('segment', F.lit('Early_Adopter'))\
                .withColumn('device_brand', F.lit('Samsung'))\
                .withColumnRenamed('Name_Temp', 'device_name')\
                .withColumnRenamed('launch', 'launch_date')
adopter = adopter.select('quarter','adoption_type','ifa','segment_id','segment','device_brand','device_name','launch_date','first_seen','delta')
adopter = adopter.cache()
adopter.select('ifa').distinct().count() # 69069
adopter.select('ifa').count() # 167058

## Early Majority
majority1 = df4.filter(col('delta').between(128,333))
majority1 = majority1.withColumn('quarter', F.lit('202109'))\
                .withColumn('adoption_type', F.lit('samsung_high_end_phones'))\
                .withColumn('segment_id', F.lit('SEG_TEL_NS_00015'))\
                .withColumn('segment', F.lit('Early_Majority'))\
                .withColumn('device_brand', F.lit('Samsung'))\
                .withColumnRenamed('Name_Temp', 'device_name')\
                .withColumnRenamed('launch', 'launch_date')
majority1 = majority1.select('quarter','adoption_type','ifa','segment_id','segment','device_brand','device_name','launch_date','first_seen','delta')
majority1 = majority1.cache()
majority1.select('ifa').distinct().count() # 212769
majority1.select('ifa').count() # 434644

## Late Majority
majority2 = df4.filter(col('delta').between(332,470))
majority2 = majority2.withColumn('quarter', F.lit('202109'))\
                .withColumn('adoption_type', F.lit('samsung_high_end_phones'))\
                .withColumn('segment_id', F.lit('SEG_TEL_NS_00016'))\
                .withColumn('segment', F.lit('Late_Majority'))\
                .withColumn('device_brand', F.lit('Samsung'))\
                .withColumnRenamed('Name_Temp', 'device_name')\
                .withColumnRenamed('launch', 'launch_date')
majority2 = majority2.select('quarter','adoption_type','ifa','segment_id','segment','device_brand','device_name','launch_date','first_seen','delta')
majority2 = majority2.cache()
majority2.select('ifa').distinct().count() # 149055
majority2.select('ifa').count() # 427771

## Laggards
laggard = df4.filter(col('delta') > 469)
laggard = laggard.withColumn('quarter', F.lit('202109'))\
                .withColumn('adoption_type', F.lit('samsung_high_end_phones'))\
                .withColumn('segment_id', F.lit('SEG_TEL_NS_00017'))\
                .withColumn('segment', F.lit('Laggard'))\
                .withColumn('device_brand', F.lit('Samsung'))\
                .withColumnRenamed('Name_Temp', 'device_name')\
                .withColumnRenamed('launch', 'launch_date')
laggard = laggard.select('quarter','adoption_type','ifa','segment_id','segment','device_brand','device_name','launch_date','first_seen','delta')
laggard = laggard.cache()
laggard.select('ifa').distinct().count() # 115926
laggard.select('ifa').count() # 194391


## Join all segments
df = innovator.union(adopter).union(majority1).union(majority2).union(laggard)
#df.select('ifa').count() # 290412
#df.select('ifa').distinct().count() # 290412
df = df.cache()
df.show(20,0)
df.printSchema()


# Write out
path = 's3a://ada-dev/DA_datamart/project_segments/adoption/samsung_high_end_phones/ID/202109'
df.write.parquet(path)


#################
