import streamlit as st
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor  # 导入回归模型
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# GBDT_data_analysis_1.py
# 设置网页标题
st.title('一个傻瓜式构建可视化 web的 Python 神器 -- streamlit')
# 展示一级标题
st.header('1. 原始数据')
df = pd.read_excel('第8组数据.xlsx')
if st.checkbox('显示原始数据集'):
    st.subheader('原始数据集')
    st.write(df)
# st.table(df)  # 芜湖
# print(df.head(5))

# 提取特征变量和目标变量
X = df.drop(columns=(['输出参数1', '序号', '输出参数2']))
y = df['输出参数1']
st.header('1. 特征变量和目标变量数据集')
if st.checkbox('显示特征变量数据集'):
    st.subheader('特征变量数据集')
    st.write(X)
if st.checkbox('显示目标变量数据集'):
    st.subheader('目标变量数据集')
    st.write(y)

# st.table(X)
# st.table(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# X_train.to_excel(r'hahaha.xlsx')
# 模型训练及搭建

model = GradientBoostingRegressor(random_state=123)
model.fit(X_train, y_train)

# 模型预测与评估
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
# 对比预测值和实际值
a = pd.DataFrame()
a['预测值'] = list(y_test_pred)
a['实际值'] = list(y_test)

st.header('1. 误差分析')
# 解释方差分
# explained_variance_score：解释方差分，这个指标用来衡量我们模型对数据集波动的解释程度，如果取值为1时，模型就完美，越小效果就越差。
st.markdown('解释方差分=%.2f' % (metrics.explained_variance_score(y_test, y_test_pred)))
# print(metrics.explained_variance_score(y_test, y_test_pred))


# 平均绝对误差
# 给定数据点的平均绝对误差，一般来说取值越小，模型的拟合效果就越好。
st.markdown('平均绝对误差=%.2f' % (metrics.mean_absolute_error(y_test, y_test_pred)))
# print(metrics.mean_absolute_error(y_test, y_test_pred))

# 均方误差
train_err = metrics.mean_squared_error(y_train, y_train_pred)
test_err = metrics.mean_squared_error(y_test, y_test_pred)
st.markdown(f"训练集误差为{train_err}")
st.markdown(f"测试集误差为{test_err}")
# print(f"训练集误差为{train_err}")
# print(f"测试集误差为{test_err}")

# 查看模型预测效果(性能评估)
score = model.score(X_test, y_test)
st.markdown(f"当前模型预测准确度为{score}")
# print(f"当前模型预测准确度为{score}")

# 可视化
X_train = np.array(X_train)  # 转换数据类型
y_train = np.array(y_train)

st.markdown(f"类型为{type(X_train)}")
# dic={
#     "特征变量(训练值)":list(X_train),
#     "值(训练值)":list(y_train)
# }
# chart_data = pd.DataFrame(dic)
# #
# st.line_chart(chart_data)

# ax_01 = plt.subplot(111)
# ax_01.spines['top'].set_visible(False)
# ax_01.spines['right'].set_visible(False)
# plt.plot(X_train, y_train, 'ro', markersize=1)
# plt.xlabel('X_train', fontsize=16)
# plt.ylabel('y_train_test', fontsize=16)
# plt.plot(X_train, y_train_pred, 'bs', markersize=1)
# plt.grid(True)
# plt.show()
# ax_02 = plt.subplot(111)
# ax_02.spines['top'].set_visible(False)
# ax_02.spines['right'].set_visible(False)
#
# X_test = np.array(X_test)
# y_test = np.array(y_test)
# plt.plot(X_test, y_test, 'ro', markersize=1)
# plt.xlabel('X_test', fontsize=16)
# plt.ylabel('y_test_pred', fontsize=16)
# plt.plot(X_test, y_test_pred, 'bs', markersize=1)
# plt.grid(True)
# plt.show()
