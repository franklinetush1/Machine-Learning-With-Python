import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')

# churn_df = pd.read_csv("ChurnData.csv")
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)

churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
 

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]  #print the first 5 valuesv

y = np.asarray(churn_df['churn'])
y[0:5] #print the first 5 values

#normalize the dataset in order to have all the features at the same scale
X_norm = StandardScaler().fit(X).transform(X)
X_norm[0:5]

X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.2, random_state=4)
LR = LogisticRegression().fit(X_train,y_train)

yhat = LR.predict(X_test)
yhat[:10]

yhat_prob = LR.predict_proba(X_test)
yhat_prob[:10]

coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()