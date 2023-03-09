import streamlit as st
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor  # 导入回归模型
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymysql

# 导入数据
@st.cache_data  # 快速缓存
def load_data(sql):
    mydb = pymysql.connect(
        host='localhost',
        user='root',
        database='hebing_1',
        passwd='asd25380'
    )
    cursor = mydb.cursor(pymysql.cursors.DictCursor)
    data_1 = []
    for i in sql:
        cursor.execute(i)
        result = cursor.fetchall()
        data_1.append(pd.DataFrame(result))
    data = data_1[0]
    for i in range(1, len(data_1)):
        data = data.join([data_1[i]], how='outer')
    return data


sql = ['select `HPC+HPT组合件相位（°）` from `1`',
       'select  `HPC+HPT组合件初始不平衡大小（gmm）` from `1`',
       'select  `HPC+HPT组合件初始不平衡角度（°）` from `1`',
       'select  `HPC+HPT组合件同心度（mm）` from `1`']
data = load_data(sql)

# 设置网页标题
st.title('基于streamlit的可视化界面')

# st.header('原始数据集')
if st.checkbox('显示原始数据集'):
    st.subheader('原始数据集')
    st.write(data)

# 获取所有列名并可视化各个列名按钮，随列的长度而变
col = [column for column in data]
col_2 = tuple(col)
genre = st.radio("请选择要预测的输出变量:", col_2)
if genre in col:
    i = col.index(genre)
    X = data.drop(columns=(col[i]))
    y = data[col[i]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    model = GradientBoostingRegressor(random_state=123)
    model.fit(X_train, y_train)
    # 模型预测与评估
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    st.header('误差分析')
    st.markdown('解释方差分误差=%.2f' % (metrics.explained_variance_score(y_test, y_test_pred)))
    st.markdown('平均绝对误差=%.2f' % (metrics.mean_absolute_error(y_test, y_test_pred)))
    train_err = metrics.mean_squared_error(y_train, y_train_pred)
    test_err = metrics.mean_squared_error(y_test, y_test_pred)
    st.markdown(f"训练集误差={train_err}")
    st.markdown(f"测试集误差={test_err}")
    # 作图
    a = pd.DataFrame()
    a['预测值'] = list(y_test_pred)
    a['实际值'] = list(y_test)
    # if st.checkbox('显示预测结果'):
    st.subheader('预测数据与实际数据对比图')
    fig, ax_01 = plt.subplots()
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    plt.plot(X_test, y_test, 'ro', markersize=1)
    plt.xlabel('X', fontsize=10)
    plt.ylabel('y', fontsize=10)
    plt.plot(X_test, y_test_pred, 'b*', markersize=1)
    plt.grid(True)
    st.pyplot(fig)

# GBDT_data_analysis_1.py

