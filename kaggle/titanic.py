# PassengerId => 乘客ID
# Pclass => 乘客等级(1/2/3等舱位)
# Name => 乘客姓名
# Sex => 性别
# Age => 年龄
# SibSp => 堂兄弟/妹个数
# Parch => 父母与小孩个数
# Ticket => 船票信息
# Fare => 票价
# Cabin => 客舱位置
# Embarked => 登船港口
# Part 1:Exploratory Data Analysis(EDA)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings

warnings.filterwarnings('ignore')
# %matplotlib inline
# data = pd.read_csv('../data/titanic/train.csv')
data = pd.read_csv('G:/kaggle/titanic/train.csv')
print(data.head())
print(data.isnull().sum())

f, ax = plt.subplots(1, 2, figsize=(18, 8))
data['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=data, ax=ax[1])
ax[1].set_title('Survived')
plt.show()

sexSurvived = data.groupby(['Sex', 'Survived'])['Survived'].count()
print(sexSurvived)

f1, ax1 = plt.subplots(1, 2, figsize=(18, 8))
data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax=ax1[0])
ax1[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue='Survived', data=data, ax=ax1[1])
ax1[1].set_title('Sex:Survived vs Dead')
plt.show()


ps = pd.crosstab(data.Pclass, data.Survived, margins=True)
print(ps)
pd.crosstab(data.Pclass, data.Survived, margins=True).style.background_gradient(cmap='summer_r')


f2, ax2 = plt.subplots(1, 2, figsize=(18, 8))
data['Pclass'].value_counts().plot.bar(color=['#CD7F32', '#FFDF00', '#D3D3D3'], ax=ax2[0])
ax2[0].set_title('Number Of Passengers By Pclass')
ax2[0].set_ylabel('Count')
sns.countplot('Pclass', hue='Survived', data=data, ax=ax2[1])
ax[1].set_title('Pclass:survived vs Dead')
plt.show()

ssp = pd.crosstab([data.Sex, data.Survived], data.Pclass, margins=True).style.background_gradient(cmap='summer_r')
print(ssp)

sns.factorplot('Pclass','Survived',hue='Sex',data=data)
plt.show()

print('Oldest Passenger was of:',data['Age'].max(),'Years')
print('Youngest Passenger was of:',data['Age'].min(),'Years')
print('Average Age on the ship:',data['Age'].mean(),'Years')

f3,ax3=plt.subplots(1,2,figsize=(18,8))
sns.violinplot('Pclass','Age',hue='Survived',data=data,split=True,ax=ax3[0])
ax3[0].set_title('Pclass and Age vs Survived')
ax3[0].set_yticks(range(0,110,10))
sns.violinplot('Sex','Age',hue='Survived',data=data,split=True,ax=ax3[1])
ax3[1].set_title('Sex and Age vs Survived')
ax3[1].set_yticks(range(0,110,10))
plt.show()

data['Initial']=0
for i in data:
    data['Initial']=data.Name.str.extract('(A-Za-z+)\.')

pd.crosstab(data.Initial, data.Sex).T.style.background_gradient(cmap='summer_r')