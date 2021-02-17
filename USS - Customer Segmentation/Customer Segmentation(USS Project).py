#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans


# In[2]:


cd C:\Users\Avinash Verma\Desktop\Data analysis


# In[3]:


data = pd.read_csv("cust.csv")
data


# In[4]:


data1 = pd.read_csv("r_freq.csv")
data1.head()


# In[5]:


data[data['Sum of Revenue']>1000000]


# In[6]:


data['Sum of Revenue'].sum()

#total revenue


# In[7]:


data[data['Jan-16']==0]


# In[8]:


data['freq']=data.iloc[:,1:13].astype(bool).sum(axis=1)
data[data['freq']==6]


# In[9]:


data['Freq by rc'] = data1['Freq by rc']
data.head()


# In[10]:


X = data[['Sum of Revenue','freq','Freq by rc']].values
X


# In[11]:


#from sklearn.preprocessing import MinMaxScaler
#scalar=MinMaxScaler()
#scalar.fit(X)
#X=scalar.transform(X)
#X


# In[12]:


km=[]
for i in range(1,11):
    k_means=KMeans(n_clusters=i)
    k_means.fit(X)
    km.append(k_means.inertia_)


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


X


# In[15]:


plt.plot(range(1,11),km)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('km')
plt.savefig("ee.png")


# In[16]:


k_means=KMeans(n_clusters=3)
y_kmeans = k_means.fit_predict(X)


# In[17]:


plt.figure(figsize=(10,5))
plt.xlabel('Total Revenue')
plt.ylabel('Frequency')
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 2], s = 1, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 2], s = 20, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 2], s = 200, c = 'g', label = 'Cluster 3')
plt.title('Clusters of customers')
plt.legend()
#plt.savefig("Customers Seg.png")


# In[18]:


#Cluster 1 (Red Color) -> low revenue
#cluster 2 (Blue Colr) -> average revenue
#cluster 3 (Green Color) -> high revenue


# In[19]:


data['Tier'] = y_kmeans


# In[20]:


#cluster 3 (Green Color) -> high revenue     Contribution - 10.66% of total revenue
print(len(data[data['Tier']==2]))
print(data[data['Tier']==2]['Sum of Revenue'].sum())


# In[21]:


#cluster 2 (Blue Colr) -> average revenue    Contribution - 28.36% of total revenue
print(len(data[data['Tier']==1]))
print(data[data['Tier']==1]['Sum of Revenue'].sum())


# In[22]:


#Cluster 1 (Red Color) -> low revenue       Contribution - 60.98% of total revenue
print(len(data[data['Tier']==0]))
print(data[data['Tier']==0]['Sum of Revenue'].sum())


# In[23]:


data[['Cust No.','freq','Freq by rc','Sum of Revenue','Tier']].to_excel("2016seg.xlsx")


# In[ ]:




