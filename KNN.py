#we have to predict class
'''
1.IMPORT DATASET

1.B PREPROCESSING

2.SPLIT 

3.TRAINING +TEST

4.X_TRAIN,Y_TRAIN

5.MODEL

6.ACCESS -CONFUSION MATRIX

7.PLOT GRAPH

p=1 manhattan
p=2 minkowski

'''
#K-NEAREST NEIGHBOURS

#IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#IMPORTING THE DATASET
dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,[2,3]].values  #independent variable x
y=dataset.iloc[:,4].values

#SPLITTING THE DATASETINTO TRAINING+TEST
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# FEATURE SCALING
#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#FITTING KNN TO TRAINING SET
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=7,metric='minkowski',p=2)
classifier.fit(X_train,y_train)

#PREDICTING THE TEST
y_pred=classifier.predict(X_test)

#MAKING CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#VISUALISING TRAINING SET RESULTS
from matplotlib.colors import ListedColormap
X_set,y_set=X_train,y_train
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
#ravel function do flattening of array
#.T is for transfering into column
#alpha is used for intensity variation lowe alpha lower intensity
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('K-NN training set')
plt.xlabel('AGE')
plt.ylabel('ESTIMATED SALARY')
plt.legend()
plt.show()

#VISUALISING TEST SET RESULTS
from matplotlib.colors import ListedColormap
X_set,y_set=X_test,y_test
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('K-NN test set')
plt.xlabel('AGE')
plt.ylabel('ESTIMATED SALARY')
plt.legend()
plt.show()
