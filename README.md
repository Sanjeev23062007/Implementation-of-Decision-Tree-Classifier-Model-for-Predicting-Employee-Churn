# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Data Preprocessing
Step 2: Train-Test Split
Step 3: Train the Model
Step 4: Predict and Evaluate

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SUDHARSANAN U
RegisterNumber: 212224230276
```
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("/content/drive/MyDrive/Employee.csv")
data.head()
```
```
data.info()
```
```
data.isnull().sum()
```
```
data["left"].value_counts()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
```
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company"]]
x.head()
```
```
y=data["left"]
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
```
```
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
```
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
*/


## Output:
![decision tree classifier model](sam.png)

![image](https://github.com/user-attachments/assets/9874bdd2-b7b3-4f2f-b3d9-89c55d881b88)

![image](https://github.com/user-attachments/assets/5c4e0a5d-f632-4164-a456-30a9a66705e5)

![image](https://github.com/user-attachments/assets/daf0461a-10ad-4dce-bcd1-4e43ccaef185)

![image](https://github.com/user-attachments/assets/603e5ded-a428-469a-8c11-be97f74209f4)

![image](https://github.com/user-attachments/assets/5598f19c-d140-40d3-8c5b-c3be67db6d59)

![image](https://github.com/user-attachments/assets/36e3b3fe-b26a-492f-accf-79f518dc4462)

![image](https://github.com/user-attachments/assets/70d4bca4-d9d6-4fef-861a-84d356e537f0)

![image](https://github.com/user-attachments/assets/380bc1b9-e600-45b3-9114-702804eb2a8e)

![image](https://github.com/user-attachments/assets/6f95b234-10fc-4eb7-bcbe-511d2ab41632)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
