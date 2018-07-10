import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn



from sklearn.datasets import load_boston
boston = load_boston()
bos = pd.DataFrame(boston.data)

boston.keys()

bos.columns = boston.feature_names
bos.head()

bos['Price']=boston.target

X=bos.iloc[:,:-1].values
y=bos.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)


from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(X_train,y_train)


y_pred=regressor.predict(X_test)

#visualizing training set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Price Predicted (Training set)')
plt.xlabel('Features')
plt.ylabel('Price')
plt.show()

plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title('Price Predicted (Test set)')
plt.xlabel('Features')
plt.ylabel('Price')
plt.show()
