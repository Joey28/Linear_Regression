
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
X_train= pd.read_csv('Linear_X_Train.csv')
Y_train=pd.read_csv('Linear_Y_Train.csv')
X_test=pd.read_csv('Linear_X_Test.csv')
Y_test=Y_train.loc[0:1249, :]

#linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#predicting test results
y_pred=regressor.predict(X_test)

# visualising the testing data
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Performance vs time(training set)')
plt.xlabel('Time')
plt.ylabel('Performance')
plt.show()

# visualising the testing data
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Performance vs time(testing set)')
plt.xlabel('Time')
plt.ylabel('Performance')
plt.show()