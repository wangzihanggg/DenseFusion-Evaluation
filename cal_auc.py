# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:25:27 2022

@author: asus
"""


import matplotlib.pyplot as plt 
import numpy as np 
from scipy.optimize import curve_fit
import scipy.integrate as si
import pandas as pd

df = pd.read_csv('./eval_result_auc_current.csv',encoding='gbk') 
print(df.columns)  #查看列名
x = df['threshold']
obj1 = df['obj1']
obj2 = df['obj2']
obj3 = df['obj3']
obj4 = df['obj4']
obj5 = df['obj5']
obj6 = df['obj6']
obj7 = df['obj7']
obj8 = df['obj8']
obj9 = df['obj9']
obj10 = df['obj10']
obj11 = df['obj11']
obj12 = df['obj12']
obj13 = df['obj13']
obj14 = df['obj14']
obj15 = df['obj15']
obj16 = df['obj16']
obj17 = df['obj17']
obj18 = df['obj18']
obj19 = df['obj19']
obj20 = df['obj20']
obj21 = df['obj21']
ALL = df['ALL']
#x=[i for i in range(25)]
#y=[14,13,13,13,13,14,15,17,19,21,22,24,27,30,31,30,28,26,24,23,21,19,17,16,14]

r1=np.polyfit(x,obj1,8)#用n次多项式拟合x，y数组
r2=np.polyfit(x,obj2,8)
r3=np.polyfit(x,obj3,8)
r4=np.polyfit(x,obj4,8)
r5=np.polyfit(x,obj5,8)
r6=np.polyfit(x,obj6,8)
r7=np.polyfit(x,obj7,8)
r8=np.polyfit(x,obj8,8)
r9=np.polyfit(x,obj9,8)
r10=np.polyfit(x,obj10,8)
r11=np.polyfit(x,obj11,8)
r12=np.polyfit(x,obj12,8)
r13=np.polyfit(x,obj13,8)
r14=np.polyfit(x,obj14,8)
r15=np.polyfit(x,obj15,8)
r16=np.polyfit(x,obj16,8)
r17=np.polyfit(x,obj17,8)
r18=np.polyfit(x,obj18,8)
r19=np.polyfit(x,obj19,8)
r20=np.polyfit(x,obj20,8)
r21=np.polyfit(x,obj21,8)
rall=np.polyfit(x,ALL,8)
f1=np.poly1d(r1)#拟合完之后用这个函数来生成多项式对象
f2=np.poly1d(r2)
f3=np.poly1d(r3)
f4=np.poly1d(r4)
f5=np.poly1d(r5)
f6=np.poly1d(r6)
f7=np.poly1d(r7)
f8=np.poly1d(r8)
f9=np.poly1d(r9)
f10=np.poly1d(r10)
f11=np.poly1d(r11)
f12=np.poly1d(r12)
f13=np.poly1d(r13)
f14=np.poly1d(r14)
f15=np.poly1d(r15)
f16=np.poly1d(r16)
f17=np.poly1d(r17)
f18=np.poly1d(r18)
f19=np.poly1d(r19)
f20=np.poly1d(r20)
f21=np.poly1d(r21)
fall=np.poly1d(rall)
#c1=f1(x)#生成多项式对象之后，就是获取x在这个多项式处的值
#plt.scatter(x,y,marker='o',label='real')#对原始数据画散点图
#plt.plot(x,c,ls='--',c='red',label='nihe')#对拟合之后的数据，也就是x，c数组画图
#plt.legend()
#plt.show()
#print(b)
# 使用函数求定积分
result=[]
result1 = si.quad(f1, 0, 0.1)  # 函数 起点 终点
result.append(result1[0]/0.1)
result2 = si.quad(f2, 0, 0.1)  # 函数 起点 终点
result.append(result2[0]/0.1)
result3 = si.quad(f3, 0, 0.1)  # 函数 起点 终点
result.append(result3[0]/0.1)
result4 = si.quad(f4, 0, 0.1)  # 函数 起点 终点
result.append(result4[0]/0.1)
result5 = si.quad(f5, 0, 0.1)  # 函数 起点 终点
result.append(result5[0]/0.1)
result6 = si.quad(f6, 0, 0.1)  # 函数 起点 终点
result.append(result6[0]/0.1)
result7 = si.quad(f7, 0, 0.1)  # 函数 起点 终点
result.append(result7[0]/0.1)
result8 = si.quad(f8, 0, 0.1)  # 函数 起点 终点
result.append(result8[0]/0.1)
result9 = si.quad(f9, 0, 0.1)  # 函数 起点 终点
result.append(result9[0]/0.1)
result10 = si.quad(f10, 0, 0.1)  # 函数 起点 终点
result.append(result10[0]/0.1)
result11 = si.quad(f11, 0, 0.1)  # 函数 起点 终点
result.append(result11[0]/0.1)
result12 = si.quad(f12, 0, 0.1)  # 函数 起点 终点
result.append(result12[0]/0.1)
result13 = si.quad(f13, 0, 0.1)  # 函数 起点 终点
result.append(result13[0]/0.1)
result14 = si.quad(f14, 0, 0.1)  # 函数 起点 终点
result.append(result14[0]/0.1)
result15 = si.quad(f15, 0, 0.1)  # 函数 起点 终点
result.append(result15[0]/0.1)
result16 = si.quad(f16, 0, 0.1)  # 函数 起点 终点
result.append(result16[0]/0.1)
result17 = si.quad(f17, 0, 0.1)  # 函数 起点 终点
result.append(result17[0]/0.1)
result18 = si.quad(f18, 0, 0.1)  # 函数 起点 终点
result.append(result18[0]/0.1)
result19 = si.quad(f19, 0, 0.1)  # 函数 起点 终点
result.append(result19[0]/0.1)
result20 = si.quad(f20, 0, 0.1)  # 函数 起点 终点
result.append(result20[0]/0.1)
result21 = si.quad(f21, 0, 0.1)  # 函数 起点 终点
result.append(result21[0]/0.1)
resultall = si.quad(fall, 0, 0.1)  # 函数 起点 终点
result.append(resultall[0]/0.1)
print(result)

aa=['obj1','obj2','obj3','obj4','obj5','obj6','obj7','obj8','obj9','obj10','obj11',
   'obj12','obj13','obj14','obj15','obj16','obj17','obj18','obj19','obj20','obj21','ALL',]

array={'Obj':pd.Series(aa),
        'AUC':pd.Series(result)}
data_auc=pd.DataFrame(array)
data_auc.to_csv('./eval_result_auc_current_result.csv',index=0,encoding="gbk")