#!/usr/bin/env python
# coding: utf-8

# <center><h1 style="font-size:35px; font-family: 'Times New Roman'; letter-spacing: 0.1em;">House Price Prediction 🏡</h1></center>

# In[1]:


import pymongo
import pandas as pd
from pymongo import MongoClient
import warnings
warnings.filterwarnings("ignore", category=Warning)


# # Reading data from Mongodb

# In[2]:


#point the client at mongo URI
client = pymongo.MongoClient("mongodb://localhost:27017/")

#select database
db = client["HPP"]

#select the collection within the database
x =db["Traindata"].find()


# In[3]:


#converting Collection into Pandas database
Df=pd.DataFrame(x)
Df.head()


# In[4]:


df=Df.drop('_id',axis=1)


# In[5]:


df.shape


# <h1 style="font-family: 'Times New Roman'; letter-spacing: 0.08em;">Importing the Essential Libraries, Metrics</h1>

# In[6]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.model_selection import GridSearchCV,cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


# <h1 style="font-family: 'Times New Roman'; letter-spacing: 0.08em;">Exploratory Data Analysis</h1>

# In[7]:


df.head()


# ***Checking the shape—i.e. size—of the data***

# In[8]:


df.shape


# ***Learning the dtypes of columns' and how many non-null values are there in those columns***

# In[9]:


df.info()


# ***Getting the statistical summary of dataset***

# In[10]:


df.describe().T


# <h1 style="font-family: 'Times New Roman'; letter-spacing: 0.08em;">Feature Selection</h1>

# ***We are selecting numerical features which have more than 0.50 or less than -0.50 correlation rate based on Pearson Correlation Method—which is the default value of parameter "method" in corr() function. As for selecting categorical features, I selected the categorical values which I believe have significant effect on the target variable such as Heating and MSZoning.***

# In[11]:


important_num_cols = list(df.corr()["SalePrice"][(df.corr()["SalePrice"]>0.50) | (df.corr()["SalePrice"]<-0.50)].index)
cat_cols = ["MSZoning", "Utilities","BldgType","Heating","KitchenQual","SaleCondition","LandSlope"]
important_cols = important_num_cols + cat_cols

df = df[important_cols]


# ***Checking for the missing values***

# In[12]:


print("Missing Values by Column")
print("-"*30)
print(df.isna().sum())
print("-"*30)
print("TOTAL MISSING VALUES:",df.isna().sum().sum())


# <h1 style="font-family: 'Times New Roman'; letter-spacing: 0.08em;">X, y Split</h1>

# ***Splitting the data into X and y chunks***

# In[13]:


X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]


# <h1 style="font-family: 'Times New Roman'; letter-spacing: 0.08em;">One-Hot Encoding</h1>

# ***Encoding the categorical features in X dataset by using One-Hot Encoding method***

# In[14]:


X = pd.get_dummies(X, columns=cat_cols)


# <h1 style="font-family: 'Times New Roman'; letter-spacing: 0.08em;">Standardizing the Data</h1>

# ***Standardizing the numerical columns in X dataset. StandardScaler() adjusts the mean of the features as 0 and standard deviation of features as 1. Formula that StandardScaler() uses is as follows:***

# In[15]:


important_num_cols.remove("SalePrice")

scaler = StandardScaler()
X[important_num_cols] = scaler.fit_transform(X[important_num_cols])


# In[16]:


X.head()


# <h1 style="font-family: 'Times New Roman'; letter-spacing: 0.08em;">Train-Test Split</h1>

# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ***Defining several evaluation functions for convenience***

# In[18]:


def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5)).mean()
    return rmse
    

def evaluation(y, predictions):
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r_squared = r2_score(y, predictions)
    return mae, mse, rmse, r_squared


# <h1 style="font-family: 'Times New Roman'; letter-spacing: 0.08em;">Machine Learning Models</h1>

# In[19]:


models = pd.DataFrame(columns=["Model","Training R2 score","Test R2 score"])


# <h2 style="font-family: 'Times New Roman'; letter-spacing: 0.05em;">Linear Regression</h2>

# In[20]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
kfold = KFold(n_splits=5,shuffle=True,random_state=2021)
parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}

LRcv = GridSearchCV(estimator=model, param_grid=parameters,cv=kfold,scoring='r2')

LRcv.fit(X_train, y_train)

print("Best Paramter :",LRcv.best_params_)
print("Training score :",LRcv.best_score_)
y_pred = LRcv.predict(X_test)
print("Test score :",r2_score(y_test, y_pred))

new_row = {"Model": "Linear","Training R2 score":LRcv.best_score_, "Test R2 score":r2_score(y_test, y_pred)}
models = models.append(new_row, ignore_index=True)


# <h2 style="font-family: 'Times New Roman'; letter-spacing: 0.05em;">Ridge Regression</h2>

# In[21]:


from sklearn.linear_model import Ridge

model = Ridge()
kfold = KFold(n_splits=5,shuffle=True,random_state=2021)
parameters = {'alpha': [1,0.1,0.01,0.001,0.0001,10]}

Rcv = GridSearchCV(estimator=model, param_grid=parameters,cv=kfold,scoring='r2')

Rcv.fit(X_train, y_train)

print("Best Paramter :",Rcv.best_params_)
print("Training score :",Rcv.best_score_)
y_pred = Rcv.predict(X_test)
print("Test score :",r2_score(y_test, y_pred))

new_row = {"Model": "Ridge","Training R2 score":Rcv.best_score_, "Test R2 score":r2_score(y_test, y_pred)}
models = models.append(new_row, ignore_index=True)


# <h2 style="font-family: 'Times New Roman'; letter-spacing: 0.05em;">Lasso Regression</h2>

# In[22]:


from sklearn.linear_model import Lasso

model = Lasso()
kfold = KFold(n_splits=5,shuffle=True,random_state=2021)
parameters ={"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]}
Lcv = GridSearchCV(estimator=model, param_grid=parameters,cv=kfold,scoring='r2')

Lcv.fit(X_train, y_train)

print("Best Paramter :",Lcv.best_params_)
print("Training score :",Lcv.best_score_)
y_pred = Lcv.predict(X_test)
print("Test score :",r2_score(y_test, y_pred))

new_row = {"Model": "Lasso","Training R2 score":Lcv.best_score_, "Test R2 score":r2_score(y_test, y_pred)}
models = models.append(new_row, ignore_index=True)


# <h2 style="font-family: 'Times New Roman'; letter-spacing: 0.05em;">Elastic Net</h2>

# In[23]:


from sklearn.linear_model import ElasticNet

model = ElasticNet()
kfold = KFold(n_splits=5,shuffle=True,random_state=2021)
parameters = {'alpha': [1,0.1,0.01,0.001,10] }

ENcv = GridSearchCV(estimator=model, param_grid=parameters,cv=kfold,scoring='r2')

ENcv.fit(X_train, y_train)

print("Best Paramter :",ENcv.best_params_)
print("Training score :",ENcv.best_score_)
y_pred = ENcv.predict(X_test)
print("Test score :",r2_score(y_test, y_pred))

new_row = {"Model": "Elastic Net","Training R2 score":ENcv.best_score_, "Test R2 score":r2_score(y_test, y_pred)}
models = models.append(new_row, ignore_index=True)


# # Decision Tree Regression

# In[24]:


from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
kfold = KFold(n_splits=5,shuffle=True,random_state=2021)
#parameters = {'max_depth':[6,8,10],'min_samples_split':[3,5,7],'min_samples_leaf':[3,4,5] }
parameters = {'max_depth':[x for x in range(10,20)],
              'min_samples_split':[x for x in range(10,16)],
              'min_samples_leaf':[x for x in range(8,16)]}
DTcv = GridSearchCV(estimator=model, param_grid=parameters,cv=kfold,scoring='r2')

DTcv.fit(X_train, y_train)

print(DTcv.best_params_)
print("Training score :",DTcv.best_score_)
y_pred = DTcv.predict(X_test)
print("Test score :",r2_score(y_test, y_pred))

new_row = {"Model": "DecisionTree","Training R2 score" :DTcv.best_score_, "Test R2 score":r2_score(y_test, y_pred)}
models = models.append(new_row, ignore_index=True)


# <h2 style="font-family: 'Times New Roman'; letter-spacing: 0.05em;">Random Forest Regressor</h2>

# In[25]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
kfold = KFold(n_splits=5,shuffle=True,random_state=2021)
parameters = {'max_depth':[5,6],'min_samples_split':[4,5],'min_samples_leaf':[4,5] }

RFcv = GridSearchCV(estimator=model, param_grid=parameters,cv=kfold,scoring='r2')

RFcv.fit(X_train, y_train)

print(RFcv.best_params_)
print("Training score :",RFcv.best_score_)
y_pred = RFcv.predict(X_test)
print("Test score :",r2_score(y_test, y_pred))

new_row = {"Model": "RandomForest","Training R2 score":RFcv.best_score_, "Test R2 score":r2_score(y_test, y_pred)}
models = models.append(new_row, ignore_index=True)


# <h2 style="font-family: 'Times New Roman'; letter-spacing: 0.05em;">KNeighbors Regressor</h2>

# In[26]:


from sklearn.neighbors import KNeighborsRegressor

parameters = {'n_neighbors': [2,4,6,8,10,12,14,16]}
print(parameters)

knn = KNeighborsRegressor()

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5 , random_state=2022,shuffle=True)

KNNcv = GridSearchCV(knn, param_grid=parameters,
                  cv=kfold,scoring='r2')
KNNcv.fit(X_train, y_train)
cv_df = pd.DataFrame(KNNcv.cv_results_  )

print(KNNcv.best_params_)
print("Training score :",KNNcv.best_score_)
y_pred = KNNcv.predict(X_test)
print("Test score :",r2_score(y_test, y_pred))

new_row = {"Model": "KNNeighbors","Training R2 score":KNNcv.best_score_, "Test R2 score":r2_score(y_test, y_pred)}
models = models.append(new_row, ignore_index=True)


# <h2 style="font-family: 'Times New Roman'; letter-spacing: 0.05em;">XGBoost Regressor</h2>

# In[27]:


from xgboost import XGBRegressor

xb=XGBRegressor(random_state=2021,use_label_encoder=False)
kfold=KFold(n_splits=5,random_state=2021,shuffle=True)
parameters={'max_depth':[5,6,7],'n_estimators':[10,15],'learning_rate':[0.4,0.5,0.6]}
XGBcv = GridSearchCV(xb, param_grid=parameters,cv=kfold,verbose=3)

XGBcv.fit(X_train, y_train)

print(XGBcv.best_params_)
print("Training score :",XGBcv.best_score_)
y_pred = XGBcv.predict(X_test)
print("Test score :",r2_score(y_test, y_pred))

new_row = {"Model": "XGBRegressor","Training R2 score":XGBcv.best_score_, "Test R2 score":r2_score(y_test, y_pred)}
models = models.append(new_row, ignore_index=True)


# <h1 style="font-family: 'Times New Roman'; letter-spacing: 0.08em;">Model Comparison</h1>

# In[33]:


a=models.sort_values(by="Test R2 score",ascending=False)
a


# In[36]:


import numpy as np
import matplotlib.pyplot as plt

X = ["XGBRegressor","RandomForest","KNNeighbors","DecisionTree","Ridge","Elastic Net","Lasso","Linear"]
Train_Score = a['Training R2 score'].head(200)
Test_score = a['Test R2 score'].head(200)

X_axis = np.arange(len(X))
width = 0.35

plt.figure(figsize=(16,6))
Train=plt.bar(X_axis - 0.2,Train_Score, 0.4, label = 'Train')
Test=plt.bar(X_axis + 0.2,Test_score, 0.4, label = 'Test')

plt.xticks(X_axis, X)
plt.xlabel("Model")
plt.ylabel("R2 Score")
plt.title("Housing Price Prediction Model Comparision")

plt.bar_label(Train, padding=3)
plt.bar_label(Test, padding=3)

plt.legend(loc='lower right')
plt.show()

