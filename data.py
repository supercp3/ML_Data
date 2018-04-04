import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
'''
数据的下载、透析和分组
'''
def data_read(path):
	data=pd.read_csv(path)
	return data
#将数据集分为训练集合和测试集合
def split_data(data,test_rate):
	shuffled=np.random.permutation(len(data))
	test_size=int(len(data)*test_rate)
	test_index=shuffled[:test_size]
	train_index=shuffled[test_size:]
	return data.iloc[train_index],data.iloc[test_index]



if __name__=="__main__":
	data_path="datasets/data/housing.csv"
	housing=data_read(data_path)
	print(housing)
	#print(housing.head(5))
	#print(housing.info())
	#print(housing['ocean_proximity'].value_counts())
	#print(housing.describe())
	#housing.hist(bins=50,figsize=(20,15))
	#plt.show()
	#housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,
	#s=housing['population']/100,label='population',c='median_house_value',cmap=plt.get_cmap("jet"),colorbar=True)
	#plt.legend()
	#plt.show()
	#数据的标准相关系数
	#corr_matrix=housing.corr()
	#corr_housing_value=corr_matrix['median_house_value'].sort_values(ascending=False)
	#print(corr_housing_value)
	#了解数据相关性的另一种方法是使用pandas的scatter_matrix函数
	#attributes=["median_house_value","median_income","housing_median_age","total_rooms"]
	#pd.scatter_matrix(housing[attributes],figsize=(12,8))
	#plt.show()
	#housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)
	#plt.show()
	#加入一些新的属性，计算相关系数
	'''
	housing['room_per_household']=housing['total_rooms']/housing['households']
	housing['bedrooms_per_room']=housing['total_bedrooms']/housing['total_rooms']
	housing['population_per_household']=housing['population']/housing['households']
	corr_matrix2=housing.corr()
	corr2=corr_matrix2['median_house_value'].sort_values(ascending=False)
	print(corr2)
	'''
	'''
	#划分训练集合和测试集合
	train_data,test_data=split_data(housing,0.2)
	print("trin_data",len(train_data),"test_data",len(test_data))
	'''
	'''
	#scikit_learn的划分数据集
	train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
	print("train_set",len(train_set),"test_set",len(test_set))
	'''
	'''
	分层抽样
	split=StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=42)
	print(type(split))
	for train_index,test_index in split.split(housing,housing['ocean_proximity']):
		strat_train_set=housing.loc[train_index]
		strat_test_set=housing.loc[test_index]
		print(len(strat_train_set))
		print(len(strat_test_set))
	xx=housing['ocean_proximity'].value_counts()/len(housing)
	print(xx)
	'''