
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import flask as flsk
from sklearn.preprocessing import StandardScaler
from pylab import rcParams
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

df = pd.read_excel("C:\\Users\91976\Downloads\customer_churn_large_dataset.xlsx")

df.head(10)

df.shape

df.info()


# Visualize missing values as a matrix
msno.matrix(df);

df.isnull().sum()

#Feature Engineering

# Removing variables that will not affect the dependent variable

df.drop(columns = ['CustomerID','Name','Location'],axis = 1,inplace = True)

df['Gender'].replace({'Male' : 1 , 'Female' : 0},inplace = True)

Gender = pd.get_dummies(df.Gender).iloc[:,1:]

df

#Modeling

X = df.drop("Churn",axis = 1)
y = df["Churn"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_st

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,max_depth=10, random_state=100)


print(X_test)

classifier.fit(X_train, y_train)


predictions = classifier.predict(X_test)

predictions


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print("Accuracy",accuracy_score(y_test, predictions))


result =confusion_matrix(y_test,predictions)
print("Confusion Matrix:")
print(result)

print("Classification Report\n",classification_report(y_test,predictions))
feat_importances = pd.Series(classifier.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)


a = float(input("Enter age :: (Range is 10 to 100 ) : "))
b = float(input("Enter Gender :: (1 for Male , 0 for Female ) : "))
c = float(input("Enter Subscription Length Moths :: (Range is 0 to 25) : "))
d = float(input("Enter Monthly Bill :: (Range is 0 to 100) "))
e = float(input("Enter Total usage of GB :: (Range is 0 to 500) : "))

data = (a,b,c,b,e)
input_Data = np.asarray(data)
input_data_reshape = input_Data.reshape(1,-1)
std_data = scaler.transform(input_data_reshape)
ans = classifier.predict(std_data)

if (ans[0] == 0):
 print('Customer will not leaft')
else:
 print('Customer will be leaft')
