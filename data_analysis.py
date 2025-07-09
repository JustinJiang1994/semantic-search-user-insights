import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS自带支持中英文
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
file_path = 'data/train.csv'
columns = ['ID', 'Age', 'Gender', 'Education', 'Query List']
df = pd.read_csv(file_path, sep='###__###', names=columns, header=None, engine='python')

# 展示基本信息
def basic_info(df):
    print('数据集基本信息:')
    print(df.info())
    print('\n前5行:')
    print(df.head())
    print('\n缺失值统计:')
    print(df.isnull().sum())

# 标签分布
def label_distribution(df):
    print('\n性别分布:')
    print(df['Gender'].value_counts())
    print('\n年龄分布:')
    print(df['Age'].value_counts().sort_index())
    print('\n学历分布:')
    print(df['Education'].value_counts())

# 查询词长度分布
def query_length_distribution(df):
    query_lens = df['Query List'].astype(str).apply(lambda x: len(x.split(',')))
    print('\n每个用户的查询词数量描述:')
    print(query_lens.describe())
    plt.hist(query_lens, bins=20, edgecolor='k')
    plt.xlabel('查询词数量')
    plt.ylabel('用户数')
    plt.title('每个用户的查询词数量分布')
    plt.show()

if __name__ == '__main__':
    basic_info(df)
    label_distribution(df)
    query_length_distribution(df) 