import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import ast
from pylab import mpl
import glob
import os
# 设置显示中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


start_time = time.time()

folder_path = '/home/hfxia/data_trans/30G_data'
# 查找文件夹下所有的 .parquet 文件
all_files = glob.glob(os.path.join(folder_path, "*.parquet"))
print(f"找到 {len(all_files)} 个 parquet 文件。")

# 读取所有 parquet 文件，并合并为一个 DataFrame
data_list = [pd.read_parquet(file) for file in all_files]
data = pd.concat(data_list, ignore_index=True)

# 1. 加载数据并去重
print(f"数据行数: {len(data)}")
data = data.drop_duplicates(subset='user_name', keep='first')
print(f"去重后数据行数: {len(data)}")

# 2. 国家与地址匹配处理
def get_country_from_address(address):
    if re.search(r'中国|北京|上海|天津|重庆|河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|河南|湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾|内蒙古|广西|西藏|宁夏|新疆|香港|澳门', str(address)):
        return '中国'
    return None

data['corrected_country'] = data['chinese_address'].apply(get_country_from_address)
pre_mask = (data['corrected_country'].notna()) & (data['country'] != data['corrected_country'])
pre_anomaly = pre_mask.sum()
data.loc[pre_mask, 'country'] = data.loc[pre_mask, 'corrected_country']
post_mask = (data['corrected_country'].notna()) & (data['country'] != data['corrected_country'])
post_anomaly = post_mask.sum()

# 异常值统计图表
plt.figure()
pd.Series({'处理前': pre_anomaly, '处理后': post_anomaly}).plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
plt.title('国家/地址不匹配处理效果')
plt.ylabel('异常数量')
plt.savefig('/home/hfxia/data_trans/30G_data/country_address_correction.png')

# 3. 年龄异常处理
age_anomalies = data[(data['age'] < 18) | (data['age'] > 70)]
print(f'年龄异常数量: {len(age_anomalies)} ({len(age_anomalies)/len(data):.2%})')

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(data['age'], bins=20, kde=True)
plt.title('处理前年龄分布')

data = data[(data['age'] >= 18) & (data['age'] <= 70)]
print(f'删除异常年龄后数据行数: {len(data)}')

plt.subplot(1, 2, 2)
sns.histplot(data['age'], bins=20, kde=True)
plt.title('处理后年龄分布')
plt.tight_layout()
plt.savefig('/home/hfxia/data_trans/30G_data/age_correction.png')

# 4. 消费历史分析
try:
    data['purchase_history'] = data['purchase_history'].apply(ast.literal_eval)
except:
    pass

data['purchase_category'] = data['purchase_history'].apply(lambda x: x.get('category', '未知'))
data['average_price'] = data['purchase_history'].apply(lambda x: x.get('average_price', 0))

# 预处理（复用之前处理结果）
price_bins = [0, 100, 500, 1000, float('inf')]
price_labels = ['0-100', '100-500', '500-1000', '1000+']
data['price_interval'] = pd.cut(data['average_price'], 
                                bins=price_bins, 
                                labels=price_labels,
                                right=False)

# 生成交叉统计表
cross_table = pd.crosstab(data['purchase_category'], 
                         data['price_interval'],
                         margins=True).sort_values('All', ascending=False)

# 筛选TOP10类别（排除"未知"和"All"）
top_categories = cross_table[(cross_table.index != '未知') & 
                            (cross_table.index != 'All')].head(10)
top_categories = top_categories.drop(columns='All')

# 可视化设置
plt.figure(figsize=(14, 8))
sns.set_palette("husl")

# 绘制堆叠柱状图
ax = top_categories.plot(kind='bar', 
                        stacked=True,
                        edgecolor='black',
                        linewidth=0.5)

# 图表美化
plt.title('TOP10商品类别的价格区间分布', fontsize=14, pad=20)
plt.xlabel('商品类别', fontsize=12)
plt.ylabel('购买人数', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加数据标签
for container in ax.containers:
    ax.bar_label(container, 
                label_type='center',
                fmt='%d',
                color='black',
                fontsize=8)

# 添加图例
plt.legend(title='价格区间', 
          bbox_to_anchor=(1.05, 1), 
          loc='upper left')

plt.tight_layout()
plt.savefig('/home/hfxia/data_trans/30G_data/purchase_category_price_distribution.png')

# 补充热力图展示
plt.figure(figsize=(12, 8))
heatmap_data = top_categories.div(top_categories.sum(axis=1), axis=0)  # 转换为百分比

sns.heatmap(heatmap_data.T, 
           annot=True, 
           fmt=".1%",
           cmap="YlGnBu",
           linewidths=0.5,
           cbar_kws={'label': '占比比例'})

plt.title('价格区间分布热力图（按类别百分比）', pad=15)
plt.xlabel('商品类别')
plt.ylabel('价格区间')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('/home/hfxia/data_trans/30G_data/purchase_category_price_heatmap.png')


age_bins = [18, 30, 40, 50, 60, 71]  # 注意右侧区间设定为闭区间或半闭区间需根据需求调整
age_labels = ['18-29', '30-39', '40-49', '50-59', '60-70']
data['age_group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels, right=False)

# 2. 统计每个年龄段中各购买类别的人数（交叉表）
purchase_by_age = pd.crosstab(data['age_group'], data['purchase_category'])
plt.figure(figsize=(12, 6))
purchase_by_age.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5)
plt.title('不同年龄段的购买类别统计')
plt.xlabel('年龄段')
plt.ylabel('购买人数')
plt.legend(title='购买类别', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('/home/hfxia/data_trans/30G_data/purchase_by_age.png')

# 3. 统计每个年龄段的平均购买价格
avg_price_by_age = data.groupby('age_group',observed=False)['average_price'].mean().round(2)
plt.figure(figsize=(8, 6))
avg_price_by_age.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('不同年龄段的平均购买价格')
plt.xlabel('年龄段')
plt.ylabel('平均价格')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('/home/hfxia/data_trans/30G_data/avg_price_by_age.png')

# 6. 信用评分分布
bins = [300, 580, 670, 740, 800, 850]
labels = ['较差(300-579)', '一般(580-669)', '良好(670-739)', '优秀(740-799)', '极好(800-850)']
data['credit_group'] = pd.cut(data['credit_score'], bins=bins, labels=labels)

# 绘制分布图
plt.figure(figsize=(12, 6))
data['credit_group'].value_counts().sort_index().plot(kind='bar', color='teal')
plt.xticks(rotation=45)
plt.xlabel('信用评分区间')
plt.ylabel('用户数量')
plt.title('信用评分分布')
plt.tight_layout()
plt.savefig('/home/hfxia/data_trans/30G_data/credit_score_distribution.png')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n数据统计和处理耗时: {elapsed_time:.2f}秒")
