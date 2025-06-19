import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df=pd.read_csv(url)

# verify successful load with some randomly selected records
df.sample(5)

df.describe()

#Select a few features that might be indicative of CO2 emission to explore more.
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.sample(9)

viz = cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
'''viz.hist()


plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
#plt.show()


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.xlim(0,27)
#plt.show()


plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='green')
plt.xlabel("Number of Cylinders")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("CYLINDERS vs CO2EMISSIONS")
plt.grid(True)
#plt.show()'''


X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# create a model object
regressor = linear_model.LinearRegression()

# train the model on the training data
# X_train is a 1-D array but sklearn models expect a 2D array as input for the training data, with shape (n_observations, n_features).
# So we need to reshape it. We can let it infer the number of observations using '-1'.
regressor.fit(X_train.reshape(-1, 1), y_train)

# Print the coefficients
print ('Coefficients: ', regressor.coef_[0]) # with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array.
print ('Intercept: ',regressor.intercept_)

plt.scatter(X_train, y_train,  color='blue')
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
#plt.show()

# Use the predict method to make test predictions
y_test_ = regressor.predict(X_test.reshape(-1,1))

'''# Evaluation
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_test_))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_test_)))
print("R2-score: %.2f" % r2_score(y_test, y_test_))'''

'''# Plot the test data and the predicted regression line
plt.scatter(X_test, y_test, color='blue', label='Actual Test Data')
plt.plot(X_test, y_test_, color='red', linewidth=2, label='Predicted Regression Line')
plt.xlabel("Engine size")
plt.ylabel("CO2 Emissions")
plt.title("Regression Model on Test Data")
plt.legend()
plt.grid(True)
plt.show()'''


# Select the feature and target variable
X_fuel = cdf[['FUELCONSUMPTION_COMB']].to_numpy()  # feature (2D array)
y_emission = cdf['CO2EMISSIONS'].to_numpy()        # target (1D array)

# Split the data
X_train_fuel, X_test_fuel, y_train_fuel, y_test_fuel = train_test_split(
    X_fuel, y_emission, test_size=0.2, random_state=42
)

