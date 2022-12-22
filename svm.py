import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("input"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv("input/train.csv") #reading the csv files using pandas
test_data = pd.read_csv("input/test.csv")

order = list(np.sort(train_data['label'].unique()))

round(train_data.drop('label', axis=1).mean(), 2)

## Separating the X and Y variable

y = train_data['label']

## Dropping the variable 'label' from X variable 
X = train_data.drop(columns = 'label')

## Normalization

X = X/255.0
test_data = test_data/255.0

# scaling the features
from sklearn.preprocessing import scale
X_scaled = scale(X)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, train_size = 0.2 ,random_state = 10)

np.set_printoptions(threshold=np.inf)
model_linear = SVC(kernel='linear')
model_linear.fit(X_train, y_train)
print(X_train.shape)
print(model_linear.coef_.shape)
print(model_linear.coef_[-1])

# predict
y_pred = model_linear.predict(X_test)
