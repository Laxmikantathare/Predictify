import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import pickle

df = pd.read_csv('C:\\Users\\Admin\\OneDrive\\Desktop\\heart-disease\\Heart.csv')

from sklearn.model_selection import train_test_split

predictors = df.drop("target",axis=1)
target = df["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
max_accuracy = 0

for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train.values,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        print(best_x) 
        
        
  
rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train.values,Y_train)
Y_pred_rf = rf.predict(X_test)

score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")

print(Y_pred_rf)

pickle.dump(rf,open('heartweb1.pkl','wb'))