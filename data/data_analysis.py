#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@ File        : /Users/justin/code/semantic-search-user-insights/data/data_analysis.py
@ Project     : /Users/justin/code/semantic-search-user-insights/data
@ Created Date: Wednesday, July 2nd 2025, 3:33:20 pm
@ Author      : Justin Jiang
@ Email       : js830910@gmail.com
@ Site        : https://github.com/JustinJiang1994
@ 
@ Copyright (c) 2025 Justin Jinag Inc.
'''
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 只读取前1000行，防止内存溢出
df = pd.read_csv('train.csv', nrows=1000)

# 展示基本信息
def show_basic_info():
    print('年龄分布:')
    print(df['Age'].value_counts())
    print('\n性别分布:')
    print(df['Gender'].value_counts())
    print('\n学历分布:')
    print(df['Education'].value_counts())

# 统计不同标签下的Query List关键词
def analyze_query_by_label(label_col, label_name_map):
    print(f'\n===== 按{label_col}分组的查询词统计 =====')
    for label, label_name in label_name_map.items():
        group = df[df[label_col] == label]
        queries = group['Query List'].dropna().str.cat(sep='|')
        words = queries.split('|')
        counter = Counter(words)
        print(f'\n{label_name}（{label}）最常见的前10个查询词:')
        print(counter.most_common(10))
        # 生成词云
        wc = WordCloud(font_path="/System/Library/Fonts/PingFang.ttc", width=800, height=400, background_color='white').generate_from_frequencies(counter)
        plt.figure(figsize=(8,4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{label_name}（{label}）词云')
        plt.show()

if __name__ == '__main__':
    show_basic_info()
    # 年龄标签映射
    age_map = {0:'未知', 1:'0-18岁', 2:'19-23岁', 3:'24-30岁', 4:'31-40岁', 5:'41-50岁', 6:'51-999岁'}
    analyze_query_by_label('Age', age_map)
    # 性别标签映射
    gender_map = {0:'未知', 1:'男性', 2:'女性'}
    analyze_query_by_label('Gender', gender_map)
    # 学历标签映射
    edu_map = {0:'未知', 1:'博士', 2:'硕士', 3:'大学生', 4:'高中', 5:'初中', 6:'小学'}
    analyze_query_by_label('Education', edu_map) 