#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicting rubber plantation yield- A regression analysis approach
Created on Fri Jul 19 16:31:09 2019
@author: Gurcan Kaynak
"""
#import numpy package for arrays and stuff 
import numpy as np 
# import matplotlib.pyplot for plotting our result 
import matplotlib.pyplot as plt 
# import pandas for importing csv files 
import pandas as pd 
from sklearn.preprocessing import Imputer
#seaborn for coloring graph
import seaborn as sns
#feature importance
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import math
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
import time
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.tree import export_graphviz
from sklearn import tree, metrics


dfYield =pd.read_csv("Data/rubber-yield.csv",sep = ",")
dfYield.columns=['Year','YieldPerHectKg']
print("The no of rows and columns are ",dfYield.shape,"respectievly")
print("Columns are: ",list(dfYield.columns)) 


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

#Master dataframe which inludes all dataframes common variables merged in together
dfMaster1=pd.merge(dfEmployees,dfProduction, on='Year',how='inner')
dfMaster2=pd.merge(dfPlantedArea,dfTapArea, on='Year',how='inner')
dfMaster3=pd.merge(dfMaster2,dfYield, on='Year',how='inner')
dfMaster=pd.merge(dfMaster1,dfMaster3, on='Year',how='inner')

print(dfMaster.describe())

X = dfMaster.iloc[:, 1:5]
y = dfMaster.iloc[:, -1]

# =============================================================================
#Taking care of missing data
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#x.values returns a numpy representation of x
imputer = imputer.fit(X.values) 
X_transformed = imputer.transform(X.values)
print(X_transformed)

# =============================================================================


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

plt.plot(AreaPlant,Yield)

plt.xlabel('AreaPlantedHect')
plt.ylabel('YieldPerHect')
    
plt.show()

plt.plot(AreaPlant,Emp)

plt.xlabel('AreaPlantedHect')
plt.ylabel('TotalPaidEmployee')
    
plt.show()
plt.plot(AreaPlant,ProduceTon)

plt.xlabel('AreaPlantedHect')
plt.ylabel('ProduceTonne')
    
plt.show()
plt.plot(AreaPlant,TapArea)

plt.xlabel('AreaPlantedHect')
plt.ylabel('TapArea')
    
plt.show()

cor1=dfPlantedArea['AreaPlantedHect'].corr(dfYield['YieldPerHectKg'])
print("correlation between plantedArea & yieldPerHect: ",cor1)
cor2=dfPlantedArea['AreaPlantedHect'].corr(dfEmployees['TotalPaidEmployee'])
print("correlation between plantedArea & TotalPaidEmployee: ",cor2)
cor3=dfPlantedArea['AreaPlantedHect'].corr(dfProduction['ProduceTonne'])
print("correlation between plantedArea & ProduceTonne: ",cor3)
cor4=dfPlantedArea['AreaPlantedHect'].corr(dfTapArea['TapAreaHect'])
print("correlation between plantedArea & TapAreaHect: ",cor4)

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

from pandas.plotting import scatter_matrix
pd.plotting.scatter_matrix(dfMaster, alpha=0.2, figsize=(10,10))
plt.show()

# Create data
N = 500
colors = (0,0,0)
area = np.pi*3

# Plot
plt.scatter(dfPlantedArea, dfYield, s=area, c=colors, alpha=0.5)
plt.title('Does Larger Plantation Area yield more rubber?')
plt.xlabel('Area Planted (in hectare)')
plt.ylabel('YIeld in Kg (per hectare)')
plt.show()

#Creating colored scattterplot graph
gapminder = dfMaster
print(gapminder.head(5))
g =sns.scatterplot(x="AreaPlantedHect", y="YieldPerHectKg",
              hue="YieldPerHectKg",
              data=gapminder);
g.set(xscale="log");
plt.show()

# skip the na values 
# find skewness in each row 
print(dfMaster.skew(axis = 0, skipna = True)) 


X = dfMaster.iloc[:,1:5]  #independent columns
y = dfMaster.iloc[:,-1]    #target column
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

y=dfMaster.YieldPerHectKg
x=dfMaster.drop('YieldPerHectKg',axis=1)
 
                     
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
 #We created a custom root mean square function that will evaluate the performance of our model.
#def RMSE (y_pred,y_test):
#    output = math.sqrt(sum((math.log(y_pred)-math.log(y_test))^2)/math.length(y_test))
#    return output 
#print(RMSE(y_pred.values,y_test.values))	 

meanSquaredError=mean_squared_error(y_test, y_pred )
print("MSE:", meanSquaredError)
rootMeanSquaredError = sqrt(meanSquaredError)
print("RMSE:", rootMeanSquaredError)
#
#for i in range(0,60):
# print("Error in value number",i,(y_test[i]-y_pred[i]))
# time.sleep(1)
#
##combined rmse value
#rss=((y_test-y_pred)**2).sum()
#mse=np.mean((y_test-y_pred)**2)
#print("Final rmse value is =",np.sqrt(np.mean((y_test-y_pred)**2)))
 
# =============================================================================
#summary=(dfMaster.describe())
#X2 = sm.add_constant(x)
#est = sm.OLS(y, X2)
#est2 = est.fit()
#print(est2.summary())
## =============================================================================


# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressorTree = DecisionTreeRegressor(random_state = 0)
regressorTree.fit(x_train, y_train)

# Predicting a new result
y_pred1 = regressorTree.predict(x_test)



print('Accuracy:',regressorTree.score(x,y))

#Visualising the Decision Tree Regression results (higher resolution)
#X_grid = np.arange(min(X), max(X), 0.01)
#X_grid = X_grid.reshape((len(X_grid), 1))
#plt.scatter(X, y, color = 'red')
#plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
#plt.title('Decision Tree Regression')
#plt.xlabel('AreaPlantedHect')
#plt.ylabel('YieldPerHectKg')
#plt.show()
#export_graphviz(regressor, out_file='tree.dot',feature_names=['YieldPerHectKg'])