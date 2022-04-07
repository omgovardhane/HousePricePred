#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark import SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.ml.feature import *
import warnings
warnings.filterwarnings("ignore")
import findspark
findspark.init()


# In[2]:


from pyspark.sql import SparkSession
from pyspark.sql import SparkSession, SQLContext
from pyspark import SparkConf, SparkContext


# In[3]:


#Clean Data Dump
inp="mongodb://127.0.0.1/Project.Housing_train"
outp="mongodb://127.0.0.1/Project.Housing_train"


# In[4]:


spark=SparkSession        .builder        .appName("Housing_pred")        .config("spark.mongodb.input.uri",inp)        .config("spark.mongodb.output.uri",outp)        .config("spark.jars.packages","org.mongodb.spark:mongo-spark-connector_2.12:2.4.2")        .getOrCreate()


# In[5]:


url_train = "https://raw.githubusercontent.com/mobassir94/Housing-price-prediction/master/train.csv"
spark.sparkContext.addFile(url_train)
df_train = spark.read.option("header","true").option("inferSchema",True).csv(SparkFiles.get("train.csv"))
df_train.show(5)
df_train.count()


# In[6]:


url_test = "https://raw.githubusercontent.com/mobassir94/Housing-price-prediction/master/test.csv"
spark.sparkContext.addFile(url_test)
df_test = spark.read.option("header","true").option("inferSchema",True).csv(SparkFiles.get("test.csv"))
df_test.show(5)
df_test.count()


# In[7]:


len(df_train.columns) omkar


# In[8]:


len(df_test.columns)


# In[9]:


df=df_train.unionByName(df_test, allowMissingColumns = True)


# In[10]:


df.show(5)


# In[11]:


df.columns


# In[12]:


df.count()


# In[13]:


len(df.columns)


# In[14]:


df.printSchema()


# # Writing Data into MongoDB

# In[7]:


df_train.write.format("com.mongodb.spark.sql.DefaultSource").option("database","Project").option("collection", "Housing_train").save()


# In[ ]:




