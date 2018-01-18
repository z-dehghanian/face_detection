
# coding: utf-8

# In[1]:


print 1


# In[2]:


1


# In[3]:


2


# In[4]:


x=[]


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA as mlabPCA
from sklearn.decomposition import PCA as sklearnPCA

mean1 = [10, 10]
mean2 = [22, 20]
cov = [[4, 4], [4, 9]]

class1 = np.random.multivariate_normal(mean1, cov, 1000).T
class2 = np.random.multivariate_normal(mean2, cov, 1000).T

plt.figure(1)

plt.plot(class1[0,:], class1[1,:], 'x')
plt.plot(class2[0,:], class2[1,:], 'x')




# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA as mlabPCA
from sklearn.decomposition import PCA as sklearnPCA

mean1 = [10, 10]
mean2 = [22, 20]
cov = [[4, 4], [4, 9]]

class1 = np.random.multivariate_normal(mean1, cov, 1000).T
class2 = np.random.multivariate_normal(mean2, cov, 1000).T

#plt.figure(1)

plt.plot(class1[0,:], class1[1,:], 'x')
plt.plot(class2[0,:], class2[1,:], 'x')

plt.show()


# In[35]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA as mlabPCA
from sklearn.decomposition import PCA as sklearnPCA

mean1 = [10, 10]
mean2 = [22, 20]
cov = [[4, 4], [4, 9]]

class1 = np.random.multivariate_normal(mean1, cov, 1000).T
class2 = np.random.multivariate_normal(mean2, cov, 1000).T

plt.plot(class1[0,:], class1[1,:], 'x')
plt.plot(class2[0,:], class2[1,:], 'x')

all_samples = np.concatenate((class1, class2), axis=1)
mlab_pca = mlabPCA(all_samples.T)

sklearn_pca = sklearnPCA(n_components=1)
sklearn_transf = sklearn_pca.fit_transform(all_samples.T)

proj = sklearn_pca.inverse_transform(sklearn_transf)
plt.plot(proj[0:1000,0],proj[0:1000,1], 'x')
plt.plot(proj[1000:2000,0],proj[1000:2000,1], 'x')

loss = ((proj - all_samples.T) ** 2).mean()
print(loss)

plt.show()

