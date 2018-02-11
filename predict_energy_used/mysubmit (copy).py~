import numpy as np 
import pandas as pd 
from sklearn import svm
from tqdm import tqdm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB							
from sklearn.gaussian_process import GaussianProcessClassifier
import csv
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier #For Classification
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


df = pd.read_csv('train.csv') #usecols = head12
df2 = pd.read_csv('test.csv') #usecols = head12

y1 = df.Energy
y2 = np.column_stack( y1 )
y3 = y2.T
y3 = y3.ravel()
#y = y2.reshape( (1,len(y2) ) )
X1 = [ 	df.T1,	df.RH_1,	df.T2,	df.RH_2,	df.T3,	df.RH_3,	df.T4,	df.RH_4,	df.T5,	df.RH_5,	df.T6	,df.RH_6,	df.T7,	df.RH_7,	df.T8,	df.RH_8,	df.T9,	df.RH_9,	df.T_out,	df.Press_mm_hg,	df.RH_out,	df.Windspeed,	df.Visibility,	df.Tdewpoint]

X2 = [ df2.T1,	df2.RH_1,	df2.T2,	df2.RH_2,	df2.T3,	df2.RH_3,	df2.T4,	df2.RH_4,	df2.T5,	df2.RH_5,	df2.T6	,df2.RH_6,	df2.T7,	df2.RH_7,	df2.T8,	df2.RH_8,	df2.T9,	df2.RH_9,	df2.T_out,	df2.Press_mm_hg,	df2.RH_out,	df2.Windspeed,	df2.Visibility,	df2.Tdewpoint]

X = np.column_stack( X1 )
Xl = np.column_stack( X2 )


#print(X.shape)
#print(y3.shape)
#clf = AdaBoostClassifier()
#clf1 =  GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
clf1 = XGBClassifier()
clf1.fit(X,y3)
clf = svm.SVC()
clf4 = tree.DecisionTreeClassifier()			#decision tree						                #simple SVM
#clf1 = GaussianNB() 					          #naive_bayes					
clf3 = GaussianProcessClassifier() 
clf = clf.fit(X,y3)
h = clf4.fit(X,y3)
clf1 = clf1.fit(X,y3)


c = clf.predict(Xl)
c1 = clf1.predict(Xl)
l = clf4.predict(X)
#c = c.ravel()
c = c.T

df3 = pd.DataFrame({"observation" : df2.Observation, "Energy" :c})

df3.to_csv("output0.csv", index=False)

#print("adaboost:")
#print(accuracy_score(y3, c, normalize=True) )
#print("xg:")
#print(accuracy_score(y3, c1, normalize=True) )
