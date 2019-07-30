#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicting rubber plantation yield- A regression analysis approach
Created on Fri Jul 19 16:31:09 2019
@author: Gurcan Kaynak
"""
#import numpy package for arrays and stuff 
import numpy as np 
#import matplotlib.pyplot for plotting our result 
import matplotlib.pyplot as plt 
#import pandas for importing csv files 
import pandas as pd 
from sklearn.preprocessing import Imputer
#seaborn for coloring graph
import seaborn as sns
#feature importance
from sklearn.ensemble import ExtraTreesClassifier
#Training and testing of our data
from sklearn.model_selection import train_test_split
#Regression Models
from sklearn.linear_model import LinearRegression
#Estimation of statistical models
import statsmodels.api as sm
#Calculating Root Mean Square Error
from sklearn.metrics import mean_squared_error
#Square root
from math import sqrt

#Exploratory Data Analysis (EDA)

#Reading our data files 
#We renamed our column headings because initial names were too long
dfYield =pd.read_csv("Data/rubber-yield.csv",sep = ",")
dfYield.columns=['Year','YieldPerHectKg']
print("The no of rows and columns are ",dfYield.shape,"respectievly")
print("Columns are: ",list(dfYield.columns)) 

#Here we used floor function to round off our values to integers because we can't have values with decimals as number of employees!
dfEmployees =pd.read_csv("Data/rubber-paidemployee.csv",sep = ",")
dfEmployees.columns=['Year','TotalPaidEmployee']
dfEmployees['TotalPaidEmployee']= dfEmployees['TotalPaidEmployee'].apply(np.floor)
print("The no of rows and columns are ",dfEmployees.shape,"respectively")
print("Columns are: ",list(dfEmployees.columns))  
 
dfProduction =pd.read_csv("Data/rubber-production.csv",sep = ",")
dfProduction.columns=['Year','ProduceTonne']
print("The no of rows and columns are ",dfProduction.shape,"respectively")
print("Columns are: ",list(dfProduction.columns)) 
 
dfPlantedArea =pd.read_csv("Data/rubber-plantedarea.csv",sep = ",")
dfPlantedArea.columns=['Year','AreaPlantedHect']
print("The no of rows and columns are ",dfPlantedArea.shape,"respectively")
print("Columns are: ",list(dfPlantedArea.columns)) 
 
dfTapArea =pd.read_csv("Data/rubber-taparea.csv",sep = ",")
dfTapArea.columns=['Year','TapAreaHect']
print("The no of rows and columns are ",dfTapArea.shape,"respectively",sep = ",")
print("Columns are: ",list(dfTapArea.columns))

#We applied inner join to merge together the dataframes because each of them had 'year' as common column
dfMaster1=pd.merge(dfEmployees,dfProduction, on='Year',how='inner')
dfMaster2=pd.merge(dfPlantedArea,dfTapArea, on='Year',how='inner')
dfMaster3=pd.merge(dfMaster2,dfYield, on='Year',how='inner')
dfMaster=pd.merge(dfMaster1,dfMaster3, on='Year',how='inner')

#Describes the dataset
print(dfMaster.describe())
print(dfMaster)


# =============================================================================
X = dfMaster.iloc[:, 0:5]
y = dfMaster.iloc[:, -1]
#Taking care of missing data
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#Train the imputor on the  dataset
imputer = imputer.fit(X.values) 
#X.values attribute return a Numpy representation of X.
#Apply the imputer to the df dataset
X_transformed = imputer.transform(X.values)
print(X_transformed)
# =============================================================================
#Data visualization: visualizing data in pursuit of finding relationship between predictors
#Drawing a box wisker plot to find the range of values in the dataset

Yield=dfMaster['YieldPerHectKg']
Emp=dfMaster['TotalPaidEmployee']
ProduceTon=dfMaster['ProduceTonne']
AreaPlant=dfMaster['AreaPlantedHect']
TapArea=dfMaster['TapAreaHect']

box_plot_data=[Yield,Emp,ProduceTon,AreaPlant,TapArea]
box=plt.boxplot(box_plot_data,patch_artist=True,labels=['Yield','Emp','ProduceTon','AreaPlant','TapArea']) 

colors = ['cyan', 'lightblue', 'lightgreen', 'tan','purple']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.show()
#==============================================================================
#Line plots to determine relationships between continuous predictors.
#YieldPerHect & PlantedArea
plt.plot(AreaPlant,Yield)

plt.xlabel('AreaPlantedHect')
plt.ylabel('YieldPerHect')
    
plt.show()
#TotalPaidEmployee & PlantedArea
plt.plot(AreaPlant,Emp)

plt.xlabel('AreaPlantedHect')
plt.ylabel('TotalPaidEmployee')
    
plt.show()
#Produce Tonne & PlantedArea
plt.plot(AreaPlant,ProduceTon)

plt.xlabel('AreaPlantedHect')
plt.ylabel('ProduceTonne')
    
plt.show()
#TapArea & Planted Area
plt.plot(AreaPlant,TapArea)

plt.xlabel('AreaPlantedHect')
plt.ylabel('TapArea')
    
plt.show()

#==============================================================================

#The evidence of strong positive linear relationship between the predictors, AreaPlantedHect, TapAreaHect, TotalPaidEmployee and ProduceTonne cannot be overlooked. 
#We, cross-check this phenomenon by deducing the correlation between them.

cor1=dfPlantedArea['AreaPlantedHect'].corr(dfYield['YieldPerHectKg'])
print("correlation between plantedArea & yieldPerHect: ",cor1)
cor2=dfPlantedArea['AreaPlantedHect'].corr(dfEmployees['TotalPaidEmployee'])
print("correlation between plantedArea & TotalPaidEmployee: ",cor2)
cor3=dfPlantedArea['AreaPlantedHect'].corr(dfProduction['ProduceTonne'])
print("correlation between plantedArea & ProduceTonne: ",cor3)
cor4=dfPlantedArea['AreaPlantedHect'].corr(dfTapArea['TapAreaHect'])
print("correlation between plantedArea & TapAreaHect: ",cor4)
#we now have ample evidence that the predictors, TotalPaidEmployee,AreaPlantedHect,ProduceTonee and TapAreaHect have a strong positive correlationship

#==============================================================================

#creating correlation plot
corr = dfMaster.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(dfMaster.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(dfMaster.columns)
ax.set_yticklabels(dfMaster.columns)
plt.show()

#==============================================================================
#Creating a scatter matrix
pd.plotting.scatter_matrix(dfMaster, alpha=0.2, figsize=(10,10))
plt.show() 
#==============================================================================
 #Does the yield increase if the plantation area increases?
 #Yield & PlantedArea
g =sns.scatterplot(x="AreaPlantedHect", y="YieldPerHectKg",
              data=dfMaster, 
              legend='full')

plt.title('Does high plantation area yield more rubber?')
plt.show()
#==============================================================================
#Data transformation


# Check for skewness of the each of the predictors
#A variable is considered 
#‘highly skewed’ if its absolute value > 1.
#‘moderately skewed’ if its absolute value > than 0.5. 
print("Check for skewness of the each of the predictors......................")
from scipy.stats import skew 
for col in dfMaster.columns: 
    column = dfMaster[col] 
    print(column.name ,':',skew(column)) 
    #column.values gives the data
   # There are no skewed predictors.
#==============================================================================
#Feature Importance we can use 3 different methods
#Method1:Univariate Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#uses the chi-squared (chi²) statistical test for non-negative features to select
#10 of the best features 
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=3)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features
#==============================================================================
X = dfMaster.iloc[:,0:5]  #independent columns
y = dfMaster.iloc[:,-1]    #target column
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
#==============================================================================
#Predictive Data Analytics

y=dfMaster.YieldPerHectKg  
x=dfMaster.drop('YieldPerHectKg',axis=1)

# Splitting the dataset into the Training set and Test set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.head()

x_train.shape
x_test.head()
x_test.shape

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)
# =============================================================================
	 
meanSquaredError=mean_squared_error(y_test, y_pred )
print("MSE:", meanSquaredError)
rootMeanSquaredError = sqrt(meanSquaredError)
print("RMSE:", rootMeanSquaredError)
print('Model Accuracy with linear regression:', regressor.score(X,y))
 
# =============================================================================
summary=(dfMaster.describe())
X2 = sm.add_constant(x)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())
#==============================================================================
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressorTree = DecisionTreeRegressor(random_state = 0)
regressorTree.fit(x_train, y_train)

# Predicting a new result
y_pred1 = regressorTree.predict(x_test)

print('Decision Tree Regression Accuracy:',regressorTree.score(x,y))

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressorForest = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressorForest.fit(x, y)

# Predicting a new result
y_pred2 = regressorForest.predict(x_test)

print('Random Forest Regression Accuracy:',regressorForest.score(x,y))
#==============================================================================

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

regressor = OLS(y, add_constant(X)).fit()
print('AIC:',regressor.aic)

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

regressor = OLS(y, add_constant(X)).fit()
print('BIC:',regressor.bic)

#Model evaluation after removing year
X1=dfMaster.iloc[:,1:5]
# Split data
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, random_state=1)

# Instantiate model
lm2 = LinearRegression()

# Fit model
lm2.fit(X1_train, y_train)

# Predict
y_prednew = lm2.predict(X1_test)

# RMSE
from math import sqrt
meanSquaredError=mean_squared_error(y_test, y_prednew )
print("MSE:", meanSquaredError)
rootMeanSquaredError = sqrt(meanSquaredError)
print("RMSE:", rootMeanSquaredError)
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
cons= OLS(y,add_constant(X1)).fit()

print("BIC:",cons.bic)
print("AIC:",cons.aic)


regressorTree.fit(X1_train,y_train)
# Predicting a new result
y_prednew1 = regressorTree.predict( X1_test)
print('Model Accuracy with model excluding year:', regressorTree.score(X1,y))
#==============================================================================
#Residual Plot Graph
#A residual plot is a graph that shows the residuals on the vertical axis and the independent variable on the horizontal axis.
#If the points in a residual plot are randomly dispersed around the horizontal axis, a linear regression model is appropriate for the data;
#otherwise, a non-linear model is more appropriate.
from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(lm2)

visualizer.fit(X1_train, y_train)  # Fit the training data to the model
visualizer.score(X1_test, y_test)  # Evaluate the model on the test data
visualizer.poof()                 # Draw/show/poof the data
#From the residual plot, we see the points are randomly distributed,
#thus the choice of our multiple linear regression was appropriate in predicting the response variable.

