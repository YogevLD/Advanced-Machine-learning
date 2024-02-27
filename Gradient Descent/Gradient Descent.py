#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 15:56:52 2023

@author: yogevladani
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
data= pd.read_csv("sample.csv")
  
###################### Gradient Descent  ######################
m_GD,c_GD,Dm_GD,Dc_GD=0,0,0,0 
l1= 0.0001
l2=0.1
epochs=1000
m_list_GD,c_list_GD,loss_list_GD=[],[],[]
x,y=data['x'],data['y']
# defining of loss function by m, c and data.
def loss_func(m,c,data):
    x=data['x']
    y=data['y']
    y_pred= m*x+c
    loss= (1/len(x))*sum(np.square(y-y_pred))
    return loss
# taking a sample of all data over 1000 timems.
for i in range(epochs):
    loss_GD = loss_func(m_GD, c_GD,data)
    y_pred= m_GD*x+c_GD
    Dm_GD = (-2/len(data))*sum(x*(y-y_pred))
    Dc_GD= (-2/len(data))*sum(y-y_pred) 
    m_GD= m_GD - l1*Dm_GD 
    c_GD= c_GD -l1*Dc_GD
    m_list_GD.append(m_GD)
    c_list_GD.append(c_GD)
    loss_list_GD.append(loss_GD)
print('The m of Gradient Descent is:',m_GD)
print('The c of Gradient Descent s:',c_GD)
print('The loss of Gradient Descent is:',loss_GD)
print('The Dm of Gradient Descent is:',Dm_GD)
print('The Dc of Gradient Descent is:',Dc_GD) 
print('The regression of Gradient Descent is:',m_GD,"X+",c_GD)

# plots of m, c and loss graph over epochs.
df_m_GD=pd.DataFrame(m_list_GD)
df_c_GD=pd.DataFrame(c_list_GD)
df_loss_GD=pd.DataFrame(loss_list_GD)
epochs_list= np.arange(1,1001)
# 3 plots one side one
fig, (m_plot, c_plot, loss_plot) = plt.subplots(1, 3, figsize=(35,15))
# create  m plot
m_plot.set_title('m plot of Gradient Descent',fontsize = 32)
m_plot.set_ylabel('m',fontsize = 25)
m_plot.set_xlabel('epochs',fontsize = 25)
m_plot.plot( df_m_GD, color = 'green')
# create  c plot
c_plot.set_title('c plot of Gradient Descent',fontsize = 32)
c_plot.set_xlabel('epochs',fontsize = 25)
c_plot.set_ylabel('c',fontsize = 25)
c_plot.plot( df_c_GD, color = 'green')
# create  loss plot
loss_plot.set_title('loss plot of Gradient Descent',fontsize = 32)
loss_plot.set_xlabel('epochs',fontsize = 25)
loss_plot.set_ylabel('loss',fontsize = 25)
loss_plot.plot( df_loss_GD, color = 'green')
plt.show()


###################### Stochastic Gradient Descent (SGD)  ######################
m_SGD,c_SGD,Dm_SGD,Dc_SGD=0,0,0,0
l1= 0.0001
l2=0.1
epochs=1000   
m_list_SGD,c_list_SGD,loss_list_SGD=[],[],[]
X,Y=data['x'],data['y']
for i in range(epochs):
    j = np.random.randint(len(data))
    x,y=X[j],Y[j]
    y_pred= m_SGD*x+c_SGD
    loss_SGD= (np.square(y-y_pred))
    Dm_SGD = (-2)*(x*(y-y_pred))
    Dc_SGD= (-2)*(y-y_pred) 
    m_SGD= m_SGD - (l1*Dm_SGD)
    c_SGD= c_SGD- (l1*Dc_SGD)
    m_list_SGD.append(m_SGD)
    c_list_SGD.append(c_SGD)
    loss_list_SGD.append(loss_SGD)
print('The m is:',m_SGD)
print('The c is:',c_SGD)
print('The loss is:',loss_SGD)
print('The Dm is:',Dm_SGD)
print('The Dc is:',Dc_SGD)
print('The regression is:',m_SGD,"X+",c_SGD)

df_m_SGD=pd.DataFrame(m_list_SGD)
df_c_SGD=pd.DataFrame(c_list_SGD)
df_loss_SGD=pd.DataFrame(loss_list_SGD)
epochs_list= np.arange(1,1001)
# 3 plots one side one
fig, (m_plot_SGD, c_plot_SGD, loss_plot_SGD) = plt.subplots(1, 3, figsize=(35,15))
# Create the m plot
m_plot_SGD.set_title('m plot of Stochastic Gradient Descent',fontsize = 32)
m_plot_SGD.set_ylabel('m',fontsize = 25)
m_plot_SGD.set_xlabel('epochs',fontsize = 25)
m_plot_SGD.plot( df_m_SGD, color = 'green')
# Create the c plot
c_plot_SGD.set_title('c plot of Stochastic Gradient Descent',fontsize = 32)
c_plot_SGD.set_xlabel('epochs',fontsize = 25)
c_plot_SGD.set_ylabel('c',fontsize = 25)
c_plot_SGD.plot( df_c_SGD, color = 'green')
# Create the loss plot
loss_plot_SGD.set_title('loss plot of Stochastic Gradient Descent',fontsize = 32)
loss_plot_SGD.set_xlabel('epochs',fontsize = 25)
loss_plot_SGD.set_ylabel('loss',fontsize = 25)
loss_plot_SGD.plot( df_loss_SGD, color = 'green')
plt.show()


###################### Mini Batch Gradient Descent ######################
m_MB,c_MB,Dm_MB,Dc_MB=0,0,0,0
l1= 0.0001
l2=0.1
epochs=1000
m_list_MB,c_list_MB,loss_list_MB=[],[],[]

for i in range((epochs)):
    random_data=data.sample(n=50)
    data_size=len(random_data)
    x,y=random_data['x'],random_data['y']
    y_pred= m_MB*x+c_MB
    loss_MB= loss_func(m=m_MB, c=c_MB, data=random_data)
    Dm_MB = (-2/data_size)*sum(x*(y-y_pred))
    Dc_MB= (-2/data_size)*sum(y-y_pred)
    m_MB= m_MB - (l1*Dm_MB)
    c_MB= c_MB- (l1*Dc_MB)
    m_list_MB.append(m_MB)
    c_list_MB.append(c_MB)
    loss_list_MB.append(loss_MB)
print('The m is:',m_MB)
print('The c is:',c_MB)
print('The loss is:',loss_MB)
print('The Dm is:',Dm_MB)
print('The Dc is:',Dc_MB)
print('The regression is:',m_MB,"X+",c_MB)

df_m_MB=pd.DataFrame(m_list_MB)
df_c_MB=pd.DataFrame(c_list_MB)
df_loss_MB=pd.DataFrame(loss_list_MB)
# 3 plots one side one
fig, (m_plot_MB, c_plot_MB, loss_plot_MB) = plt.subplots(1, 3, figsize=(35,15))
# Create the m plot
m_plot_MB.set_title('m plot of Mini Batch Gradient Descent',fontsize = 32)
m_plot_MB.set_ylabel('m',fontsize = 25)
m_plot_MB.set_xlabel('epochs',fontsize = 25)
m_plot_MB.plot( df_m_MB, color = 'green')
# Create the c plot
c_plot_MB.set_title('c plot of SMini Batch Gradient Descent',fontsize = 32)
c_plot_MB.set_xlabel('epochs',fontsize = 25)
c_plot_MB.set_ylabel('c',fontsize = 25)
c_plot_MB.plot( df_c_MB, color = 'green')
# Create the loss plot
loss_plot_MB.set_title('loss plot of Mini Batch Gradient Descent',fontsize = 32)
loss_plot_MB.set_xlabel('epochs',fontsize = 25)
loss_plot_MB.set_ylabel('loss',fontsize = 25)
loss_plot_MB.plot( df_loss_MB, color = 'green')
plt.show()
