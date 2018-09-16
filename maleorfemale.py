
# coding: utf-8

# In[66]:


import numpy as np
from sklearn import tree
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# In[67]:


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# In[68]:


clf_tree=tree.DecisionTreeClassifier()
clf_svm=svm.LinearSVC()
clf_rendomforest=RandomForestClassifier()


# In[69]:


clf_tree.fit(X,Y)
clf_svm.fit(X,Y)
clf_rendomforest.fit(X,Y)


# In[70]:


_X=[[184,84,44],[198,92,48],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
_Y=['male','male','male','female','female','female','male','male']


# In[71]:


pred_tree=clf_tree.predict(_X)
pred_svm=clf_svm.predict(_X)
pred_randomforest=clf_rendomforest.predict(_X)


# In[72]:


accuracy_tree=accuracy_score(_Y,pred_tree)*100
print("acc of decision tree:{}".format(accuracy_tree))
accuracy_svm=accuracy_score(_Y,pred_svm)*100
print("acc of svm:{}".format(accuracy_tree))
accuracy_randomforest=accuracy_score(_Y,pred_randomforest)*100
print("acc of random forest:{}".format(accuracy_randomforest))


# In[80]:


from sklearn import neighbors
clf_KNN=neighbors.KNeighborsClassifier()
clf_KNN.fit(X,Y)
pred_KNN=clf_KNN.predict(_X)
print(pred_KNN)
accuracy_KNN=accuracy_score(_Y,pred_KNN)*100


# In[81]:


np.argmax([accuracy_tree,accuracy_svm,accuracy_randomforest,accuracy_KNN])
classifier={0:'tree',1:'svm',2:'randomforest'}
print("Best classifier is :{} ".format(classifier))

