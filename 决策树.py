# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:07:52 2020

@author: HP
"""


import pandas as pd
from sklearn import model_selection,metrics,tree,svm,preprocessing
import time
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
import matplotlib.pyplot as plt


dt0=pd.read_csv('D:\\wine.data',header=None)

y=dt0[0]
x=dt0.loc[:,1:13]
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.25,random_state=1)


starttime=time.time()
clf=tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
score1=clf.score(x_test,y_test)
predicted_y=clf.predict(x_test)
hunxiaojuzhen1=metrics.confusion_matrix(y_test,predicted_y)
report1=metrics.classification_report(y_test,predicted_y)
endtime=time.time()
usedtime0=endtime-starttime
print('决策树效果：用时{}秒，\n得分{}，\n性能报告如下：\n'.format(usedtime0,score1),report1)



x_scale=preprocessing.scale(x) #把数据集标准化
x_train,x_test,y_train,y_test=model_selection.train_test_split(x_scale,y,test_size=0.3,random_state=1)

starttime=time.time()
clf2=svm.SVC(kernel='rbf',C=100,gamma=0.6)
clf2.fit(x_train,y_train)
score2=clf2.score(x_test,y_test)
predicted_y=clf.predict(x_test)
report2=metrics.classification_report(y_test,predicted_y)
endtime=time.time()
usedtime1=endtime-starttime
print('SVM效果：用时{}秒，\n得分{}，\n性能报告如下：\n'.format(usedtime1,score2),report2)



x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.25,random_state=1)
# 以下代码定义模型结构
model1=Sequential()
model1.add(Dense(units=39,input_shape=(13,)))
model1.add(Activation('relu'))
model1.add(Dense(39))
model1.add(Activation('relu'))
model1.add(Dense(3))
model1.add(Activation('softmax'))
#定义损失函数和优化器并编译
model1.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
#进行多分类前应先把无意义的、超过2个取值的变量进行独热处理
y_train_one_hot=pd.get_dummies(y_train,)
y_test_one_hot=pd.get_dummies(y_test,)
#在keras上训练模型
starttime=time.time()
model1.fit(x_train,y_train_one_hot,epochs=350,batch_size=1,verbose=2,validation_data=(x_test,y_test_one_hot))
endtime=time.time()
usedtime2=endtime-starttime
#评估模型
loss,accuracy=model1.evaluate(x_test,y_test_one_hot,verbose=2)
print('keras训练模型的效果：loss:{}，accuracy：{}，用时{}秒\n'.format(loss,accuracy,usedtime2))
classes=model1.predict(x_test,batch_size=1,verbose=2)
print('测试样本数：',len(classes))
print('分类概率：',classes)

times=pd.Series([usedtime0,usedtime1,usedtime2],)
scores=pd.Series([score1,score2,accuracy])
times.index=['sklearn_tree','SVM','Neural Networks']
scores.index=['sklearn_tree','SVM','Neural Networks']
times.plot(kind='bar',rot=0,title='times')
plt.show()
scores.plot(kind='bar',rot=0,title='scores')
plt.show()
print('用时：\n',times)


