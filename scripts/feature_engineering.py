import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# 配置
DATA_DIR = 'data'
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
SEP = '###__###'
COL_NAMES = ["ID", "Age", "Gender", "Education", "Query List"]

# 读取数据
def load_data(file, nrows=None):
    return pd.read_csv(file, sep=SEP, header=None, names=COL_NAMES, engine='python', nrows=nrows)

def preprocess_query_list(df):
    # 统计特征
    df['query_count'] = df['Query List'].fillna('').apply(lambda x: len([w for w in x.split('|') if w.strip()]))
    df['unique_query_count'] = df['Query List'].fillna('').apply(lambda x: len(set([w for w in x.split('|') if w.strip()])))
    # 用空格连接，便于TfidfVectorizer
    df['query_str'] = df['Query List'].fillna('').apply(lambda x: ' '.join([w for w in x.split('|') if w.strip()]))
    return df

def extract_tfidf_feature(train_df, test_df, max_features=500):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_df['query_str'])
    X_test = vectorizer.transform(test_df['query_str'])
    tfidf_cols = [f'tfidf_{i}' for i in range(X_train.shape[1])]
    train_tfidf = pd.DataFrame(X_train.toarray(), columns=tfidf_cols, index=train_df.index)
    test_tfidf = pd.DataFrame(X_test.toarray(), columns=tfidf_cols, index=test_df.index)
    return train_tfidf, test_tfidf

def encode_labels(df, label_cols):
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def main():
    print('读取数据...')
    train_df = load_data(TRAIN_FILE)
    test_df = load_data(TEST_FILE)
    print('处理Query List和统计特征...')
    train_df = preprocess_query_list(train_df)
    test_df = preprocess_query_list(test_df)
    print('提取TF-IDF特征...')
    train_tfidf, test_tfidf = extract_tfidf_feature(train_df, test_df, max_features=500)
    print('标签编码...')
    train_df = encode_labels(train_df, ['Age', 'Gender', 'Education'])
    # 合并特征
    train_features = pd.concat([train_df[['ID', 'Age', 'Gender', 'Education', 'query_count', 'unique_query_count']], train_tfidf], axis=1)
    test_features = pd.concat([test_df[['ID', 'query_count', 'unique_query_count']], test_tfidf], axis=1)
    # 保存
    print('保存特征数据...')
    train_features.to_csv(os.path.join(DATA_DIR, 'train_features.csv'), index=False)
    test_features.to_csv(os.path.join(DATA_DIR, 'test_features.csv'), index=False)
    print('特征工程完成！')

if __name__ == '__main__':
    main() 