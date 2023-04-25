import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
import kaplanmeier as km
%matplotlib inline
plt.rcParams["figure.figsize"] = (15,10)

df_train, df_test = pd.read_csv("training.txt", header=None), pd.read_csv("test.txt", header=None)
df_train.columns, df_test.columns = ["Cenus", "Ac", "y", "stime"], ["Cenus", "Ac", "y", "stime"]
print(df_train.head(), df_test.head(), sep="\n")


#%%%%%%%%%%% KERNEL FOR TRACKING DATA (3D nucleus ellipsoid shape, 3D nucleus volume) in Fig. 4 A) %%%%%%%%%%%%%%%
def kernel(X, Y):
    '''Custom sigmoid kernel'''
    M = np.array([[1, .1], [0.6, 1.9]]) # rotation
    alpha, b, c = .04, 5, .9  # hyper-parameters
    f = lambda x: np.tanh(x) 
    
    return f(alpha*np.dot(np.dot(X, M), Y.T)+c)*b
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%% KERNEL METASTASIS CLASSIFICATION (CeNuS, mean standardized cell area) in Fig. 6 %%%%%%%%%%%%%%%
def kernel(X, Y):
    '''Custom sigmoid kernel'''
    M = np.array([[1, .1], [0., 1.75]]) # rotation
    alpha, b, c = .0005, 10., .75  # hyper-parameters
    f = lambda x: np.tanh(x) # general sigmoid :) 
    
    return f(alpha*np.dot(np.dot(X, M), Y.T)+c)*b
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%% KERNEL METASTASIS CLASSIFICATION (Cell Shape, mean standardized cell area) in Fig. 13 %%%%%%%%%%%%%%%
def kernel(X, Y):
    '''Custom sigmoid kernel'''
    M = np.array([[1, .1], [-0.2, 1.75]]) # rotation
    alpha, b, c, d = .0005, 19., 0.65, 1.5  # hyper-parameters
    f = lambda x: np.tanh(x) # general sigmoid :) 
    
    return f(alpha*np.dot(np.dot(X, M), Y.T)+c)*b - d
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%% KERNEL METASTASIS CLASSIFICATION (Nucleus AR, mean standardized cell area) in Fig. 12 %%%%%%%%%%%%%%%
def kernel(X, Y):
    '''Custom sigmoid kernel'''
    M = np.array([[1, .1], [-0.05, 1.75]]) # rotation
    alpha, b, c = .0003, 13., .65  # hyper-parameters (original: alpha, b, c = .0005, 10., .75)
    f = lambda x: np.tanh(x) # general sigmoid :) 
    
    return f(alpha*np.dot(np.dot(X, M), Y.T)+c)*b
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




def model(df,kernel): return SVC(C=5, kernel=kernel, class_weight="balanced").fit(df_train.iloc[:,:2], df_train.iloc[:,2])

classifier =  model(df_train, kernel)

def decision_boundary(classifier, df, show=None):

    disp = DecisionBoundaryDisplay.from_estimator(
    classifier, df.iloc[:,:2], response_method="predict",
    xlabel="Cenus", 
    ylabel="Ac",
    alpha=0.2)
    
    if show:
        disp.ax_.scatter(df.Cenus[df.y==0], df.Ac[df.y==0], s=df.y[df.y==0]+100, 
            marker='o',
            color='blue',
            label='Control', alpha=0.2);
        plt.scatter(df.Cenus[df.y==1], df.Ac[df.y==1], s=df.y[df.y==1]+100, 
            marker='d',
            color='red',
            label='Case', alpha=0.8);
        plt.show();
    return disp 

db = decision_boundary(classifier,df_train, 1)

_ = decision_boundary(classifier,df_test, 1)

df_train["pred"] = classifier.predict(df_train.iloc[:,:2])
df_test["pred"] = classifier.predict(df_test.iloc[:,:2])

km_train = km.fit(df_train.stime, df_train.y, df_train.pred)
km_test =  km.fit(df_test.stime, df_test.y, df_test.pred)

km.plot(km_train)

km.plot(km_test)


