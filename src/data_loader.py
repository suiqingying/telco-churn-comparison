import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    # 假设数据文件名为 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # 删除 customerID
    df = df.drop('customerID', axis=1)

    # 处理 TotalCharges 字段中的空值
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # 标签编码
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    # 分离特征和标签
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 独热编码（只对object类型的特征）
    X = pd.get_dummies(X)

    # 数值特征归一化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练集、测试集（8:2）
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    return (X_train, y_train.values), (X_val, y_val.values), (X_test, y_test.values)