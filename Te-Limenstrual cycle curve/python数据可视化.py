#!/usr/bin/env python
# coding: utf-8

# In[147]:


import matplotlib.pyplot as plt


# In[148]:


import numpy as np


# In[149]:


x = np.arange(0,1,0.05)
print(x)


# In[150]:


# y = sin(2*pi*x)
y = np.sin(2*np.pi*x)
print(y)


# In[151]:


plt.title('sin function')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,y,'p--',label='sin')
plt.legend(loc='best')
plt.show()


# In[152]:


fig,ax=plt.subplots(2,2)
ax[0,1].plot(x,y)
plt.show()


# In[153]:


# y = cos(2*pi*x)
y1 = np.cos(2*np.pi*x)
print(y1)


# In[164]:


fig,ax=plt.subplots()
ax.plot(x,y,'r--',label='sin function')
ax.plot(x,y1,'g--o',label='cos function')
ax.set(title='curve function')
ax.set(xlabel='x')
ax.set(ylabel='y')
legend=ax.legend(loc='best')
plt.show()


# In[165]:


## 保存图片
fig.savefig('math_function.png')


# In[ ]:




