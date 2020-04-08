#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys
print('Python: ()'.format(sys.version))
import scipy
print('scipy: ()'.format(scipy.__version__))
import numpy
print('numpy: ()'.format(numpy.__version__))
import matplotlib
print('matplotlib: ()'.format(matplotlib.__version__))
import pandas
print('pandas: ()'.format(pandas.__version__))
import sklearn
print('sklearn: ()'.format(sklearn.__version__))


# In[6]:


import sys
print('Python: {}'.format(sys.version))
import scipy
print('scipy: {}'.format(scipy.__version__))
import numpy
print('numpy: {}'.format(numpy.__version__))
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
import pandas
print('pandas: {}'.format(pandas.__version__))
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# In[13]:


#load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[15]:


data = pd.read_csv("Desktop/IRIS.csv")


# In[16]:


print(data.shape)


# In[17]:


print(data.head(30))


# In[19]:


print(data.describe())


# In[21]:


print(data.groupby('species').size())


# In[30]:


data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False )


# In[31]:


data.hist()
plt.show()


# In[32]:


scatter_matrix(data)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




