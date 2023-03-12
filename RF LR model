##导入基础包
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False
import seaborn as sns
from sklearn import preprocessing


##导入文件并合并
df_PLAYER=pd.read_csv(r'/Users/tanshuyin/Desktop/PLAYER.csv')
df_HOF1=pd.read_csv(r'/Users/tanshuyin/Desktop/HOF1.csv')
df=pd.concat([df_PLAYER,df_HOF1])
df.head()

df.dropna(thresh = 31)
##去重
df.drop_duplicates()
df=df.reset_index(drop=True)

##删去明显无关的特征或相关性较强特征
df.drop(['player_id','player','games'],axis=1,inplace=True)

#查看正负样本占比
pos_data=df[df.hof==1]
neg_data=df[df.hof==0]
print('正样本数量：{}，所占比例：{}'.format(len(pos_data),len(pos_data)/len(df)))
print('负样本数量：{}，所占比例：{}'.format(len(neg_data),len(neg_data)/len(df)))

##特征分类
##数值型特征
num_cols=['num_seasons','points','assists','rebound','blocks','steals','minutes','FG','FG%','eFG%'\
    ,'All NBA 1st team','All NBA 2nd team','All NBA 3rd team','All Defense 1st team','All Defense 2nd team'\
    ,'All Rookie 1st team','All Rookie 2nd team','All ABA 1st team','All ABA 2nd team','All Star appearances'\
    ,'MVPs','DPOY','NBA ROY','MIP','SMOY','ABA MVP','ABA ROY']


##目标类别
target_cols=['hof']

## 所有特征列
total_cols=num_cols
used_data=df[total_cols+target_cols]

#print('使用{}列数据作为特征'.format(len(total_cols)))


used_num_feats=used_data[num_cols].values
used_feats=np.hstack((used_num_feats,))
used_target=used_data[target_cols].values
used_feats=pd.DataFrame(used_feats)
used_target=pd.DataFrame(used_target)
used_target.columns=['hof']

used_data=pd.concat([used_feats,used_target],axis=1)
used_data.head()





## 分割训练集，测试集
##手动分割

'''pos_data=used_data[used_data['hof']==1].reindex()
train_pos_data=pos_data.iloc[:int(len(pos_data)*0.75)].copy()
test_pos_data=pos_data.iloc[int(len(pos_data)*0.75):].copy()


neg_data=used_data[used_data['hof']==0].reindex()
train_neg_data=neg_data.iloc[:int(len(neg_data)*0.75)].copy()
test_neg_data=neg_data.iloc[int(len(neg_data)* 0.75):].copy()

train_data=pd.concat([train_pos_data,train_neg_data])
test_data=pd.concat([test_pos_data,test_neg_data])

print('训练集数据个数',len(train_data))
print('正负样本比例',len(train_pos_data)/len(train_neg_data))
print('测试集数据个数',len(test_data))
print('正负样本比例',len(test_pos_data)/len(test_neg_data))
train_feats=train_data.iloc[:,:-1].values
test_feats=test_data.iloc[:,:-1].values

train_target=train_data[['hof']].values
test_target=test_data[['hof']].values

print('训练数据:',train_feats.shape)
print('测试数据:',test_feats.shape)

print('训练标签:',train_target.shape)
print('测试标签:',test_target.shape)'''

#使用sklearn自带方法分割
##分割训练、测试集
from sklearn.model_selection import train_test_split
X = df.drop(['hof'], axis=1)
y = df['hof']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


#重采样
from imblearn.over_sampling import SMOTE

print('重采样前：')
print('正样本数：',len(y_train[y_train==1]))
print('负样本数：',len(y_train[y_train==0]))

sm=SMOTE(random_state=0)
X_re_train,y_re_train=sm.fit_resample(X_train.values,y_train.values)

print('重采样后：')
print('正样本数：',len(y_re_train[y_re_train==1]))
print('负样本数：',len(y_re_train[y_re_train==0]))


#最终使用数据
print(len(X_re_train))
print(len(y_re_train))
print(len(X_test))
print(len(y_test))


#LR模型 
##逻辑回归需要进行标准化，因为数据本身之间量纲存在差距，
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_test = sc_X.fit_transform(X_test)
X_re_train = sc_X.fit_transform(X_re_train)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(penalty='l2',solver='newton-cg')
print(LR.fit(X_re_train, y_re_train))
print("训练集准确率: ", LR.score(X_re_train,y_re_train))
print("测试集准确率: ", LR.score(X_test,y_test))
#若pennalty金使用l1 则仅可使用liblinear和saga 若使用l2则全部都可使用
# liblinear：坐标轴下降法来迭代优化损失函数。
# lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
# newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
# sag：随机平均梯度下降，梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅用一部分的样本来计算梯度，适合于样本数据多的时候。
# saga：快速梯度下降法，线性收敛的随机优化算法的的变种。

from sklearn import metrics
X_train_pred =LR.predict(X_re_train)
X_test_pred = LR.predict(X_test)
print('训练集混淆矩阵:')
print(metrics.confusion_matrix(y_re_train, X_train_pred,labels=[1,0]))
print('测试集混淆矩阵:')
print(metrics.confusion_matrix(y_test, X_test_pred,labels=[1,0]))

##输出矩阵

from sklearn.metrics import classification_report
print('训练集:')
print(classification_report(y_re_train, X_train_pred))
print('测试集:')
print(classification_report(y_test, X_test_pred))'''

#随机森林
#寻找最优参数
#这里找到的最优参数n_estimators为41

'''from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

param_test1={'n_estimators':range(1,101,10)}
gsearch=GridSearchCV(estimator=RandomForestClassifier(),param_grid=param_test1,
                    scoring='roc_auc',cv=10)
gsearch.fit(X_re_train,y_re_train)

print('最优选{}'.format(gsearch.best_params_))
print('最优选准确率{}'.format(gsearch.best_score_))'''

from sklearn.decomposition import PCA
pca=PCA(n_components=27)
X_re_train_rf=pca.fit_transform(X_re_train)


X_test_rf=pca.fit_transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

rf_clf=RandomForestClassifier(random_state=0,n_estimators=41)
print(rf_clf.fit(X_re_train,y_re_train))
print('训练集准确率:',rf_clf.score(X_re_train,y_re_train))


X_train_pred = rf_clf.predict(X_re_train)
X_test_pred = rf_clf.predict(X_test)

print('训练集混淆矩阵:')
print(metrics.confusion_matrix(y_re_train,X_train_pred,labels=[1,0]))
print('测试集混淆矩阵:')
print(metrics.confusion_matrix(y_test,X_test_pred,labels=[1,0]))

from sklearn.metrics import classification_report
print('训练集:')
print(classification_report(y_re_train, X_train_pred))
print('测试集:')
print(classification_report(y_test, X_test_pred))
