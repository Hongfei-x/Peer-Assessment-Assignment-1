import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from pylab import mpl
import glob
import os
from collections import Counter

# 设置显示中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

start_time = time.time()
folder_path = '/home/hfxia/data_trans/10G_data_new'
all_files = glob.glob(os.path.join(folder_path, "*.parquet"))
print(f"找到 {len(all_files)} 个 parquet 文件。")

# 只读取必要列以节省内存
use_cols = [
    'user_name', 'age', 'gender', 'address', 'country',
    'purchase_history', 'login_history'
]

# 全局统计变量
raw_total_rows = 0
seen_users = set()
total_users = 0

# 计数器
age_pass_count = 0
gender_counter = Counter()
abnormal_address_counter = 0
country_counter = Counter()
product_counter = {}
payment_counter = {}
device_counter = {}

age_product = {}
age_payment = {}
gender_product = {}
gender_payment = {}

def safe_load_json(s):
    try:
        return json.loads(s) if pd.notnull(s) else {}
    except:
        return {}

def age_group_func(age):
    if 18 <= age <= 35:
        return '青年'
    elif 36 <= age <= 55:
        return '中年'
    elif 56 <= age <= 70:
        return '老年'
    else:
        return '其它'

# 增量读取、去重并统计
for file in all_files:
    df_chunk = pd.read_parquet(file, columns=use_cols)
    raw_total_rows += len(df_chunk)
    # 仅保留第一次见到的 user_name
    mask_new = ~df_chunk['user_name'].isin(seen_users)
    df_chunk = df_chunk[mask_new]
    seen_users.update(df_chunk['user_name'])
    total_users += len(df_chunk)

    # 年龄在18-70统计
    ages = df_chunk['age']
    age_pass_count += ((ages >= 18) & (ages <= 70)).sum()

    # 性别计数
    gender_counter.update(df_chunk['gender'].astype(str))

    # 地址异常计数
    abnormal_address_counter += (df_chunk['address'] == "Non-Chinese Address Placeholder").sum()

    # 国家字段更新后的计数
    for addr, orig_cty in zip(df_chunk['address'], df_chunk['country']):
        if addr != "Non-Chinese Address Placeholder":
            country_counter["中国"] += 1
        else:
            country_counter[orig_cty] += 1

    # 购买历史和登录历史解析、各项统计
    for _, row in df_chunk.iterrows():
        rec = safe_load_json(row['purchase_history'])
        cat = rec.get("categories")
        pay = rec.get("payment_method")
        if cat:
            product_counter[cat] = product_counter.get(cat, 0) + 1
        if pay:
            payment_counter[pay] = payment_counter.get(pay, 0) + 1

        # 按年龄组统计
        ag = age_group_func(row['age'])
        if ag in ('青年', '中年', '老年'):
            age_product.setdefault(ag, {})
            age_payment.setdefault(ag, {})
            if cat:
                age_product[ag][cat] = age_product[ag].get(cat, 0) + 1
            if pay:
                age_payment[ag][pay] = age_payment[ag].get(pay, 0) + 1

        # 按性别（男/女）统计
        g = row['gender']
        if g in ('男', '女'):
            gender_product.setdefault(g, {})
            gender_payment.setdefault(g, {})
            if cat:
                gender_product[g][cat] = gender_product[g].get(cat, 0) + 1
            if pay:
                gender_payment[g][pay] = gender_payment[g].get(pay, 0) + 1

        # 登录设备统计
        rec2 = safe_load_json(row['login_history'])
        devices = rec2.get("devices", [])
        if isinstance(devices, list):
            for d in devices:
                device_counter[d] = device_counter.get(d, 0) + 1

# 输出去重前后行数
print(f"数据行数: {raw_total_rows}")
print(f"去重后数据行数: {total_users}")

#############################
# 任务1：筛选出年龄在18-70岁的用户，并统计比例
#############################
ratio_age = age_pass_count / total_users
print("年龄在18-70岁用户比例：{:.2%}".format(ratio_age))

counts_age = [age_pass_count, total_users - age_pass_count]
labels_age = ['年龄在18-70', '其他']
plt.figure(figsize=(6,6))
plt.pie(counts_age, labels=labels_age, autopct='%1.1f%%', startangle=90)
plt.title("年龄筛选（18-70岁）用户比例")
plt.savefig("10G_data_new/age_filter_ratio.png")
plt.show()

#############################
# 任务2：统计性别比例（注意性别可能不止男、女）
#############################
gender_counts = pd.Series(gender_counter)
print("性别统计：")
print(gender_counts)

plt.figure(figsize=(6,6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("用户性别比例")
plt.savefig("10G_data_new/gender_ratio.png")
plt.show()

#############################
# 任务3：地址异常统计
#############################
print("地址异常用户数量：", abnormal_address_counter)
counts_addr = [abnormal_address_counter, total_users - abnormal_address_counter]
labels_addr = ['地址异常', '地址正常']
plt.figure(figsize=(6,6))
plt.pie(counts_addr, labels=labels_addr, autopct='%1.1f%%', startangle=90)
plt.title("地址异常比例")
plt.savefig("10G_data_new/abnormal_address_ratio.png")
plt.show()

#############################
# 任务4：更新 country 并统计
#############################
country_counts = pd.Series(country_counter)
print("国家字段统计：")
print(country_counts)

plt.figure(figsize=(8,6))
country_counts.plot(kind='bar')
plt.xlabel("国家")
plt.ylabel("用户数量")
plt.title("更新后各国家用户分布（地址在中国更新为中国）")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("10G_data_new/country_distribution.png")
plt.show()

#############################
# 任务5：购买商品及支付方式统计
#############################
print("不同商品统计：", product_counter)
print("不同支付方式统计：", payment_counter)

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.bar(product_counter.keys(), product_counter.values())
plt.xlabel("商品类别")
plt.ylabel("数量")
plt.title("购买商品分布")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.bar(payment_counter.keys(), payment_counter.values())
plt.xlabel("支付方式")
plt.ylabel("数量")
plt.title("支付方式分布")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("10G_data_new/purchase_product_payment_distribution.png")
plt.show()

#############################
# 任务6：不同年龄段购买与支付分布
#############################
plt.figure(figsize=(12,5))
for grp in sorted(age_product):
    products = age_product[grp]
    plt.bar([f"{grp}-{p}" for p in products], products.values(), label=grp)
plt.xlabel("年龄组-商品类别")
plt.ylabel("数量")
plt.title("不同年龄段购买商品分布")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("10G_data_new/agegroup_product_distribution.png")
plt.show()

plt.figure(figsize=(12,5))
for grp in sorted(age_payment):
    pays = age_payment[grp]
    plt.bar([f"{grp}-{m}" for m in pays], pays.values(), label=grp)
plt.xlabel("年龄组-支付方式")
plt.ylabel("数量")
plt.title("不同年龄段支付方式分布")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("10G_data_new/agegroup_payment_distribution.png")
plt.show()

#############################
# 任务7：不同性别购买与支付分布
#############################
plt.figure(figsize=(12,5))
for g in gender_product:
    prods = gender_product[g]
    plt.bar([f"{g}-{p}" for p in prods], prods.values(), label=g)
plt.xlabel("性别-商品类别")
plt.ylabel("数量")
plt.title("不同性别购买商品分布")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("10G_data_new/gender_product_distribution.png")
plt.show()

plt.figure(figsize=(12,5))
for g in gender_payment:
    pays = gender_payment[g]
    plt.bar([f"{g}-{m}" for m in pays], pays.values(), label=g)
plt.xlabel("性别-支付方式")
plt.ylabel("数量")
plt.title("不同性别支付方式分布")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("10G_data_new/gender_payment_distribution.png")
plt.show()

#############################
# 任务8：登录历史设备分布
#############################
print("设备使用统计：", device_counter)
plt.figure(figsize=(8,6))
plt.bar(device_counter.keys(), device_counter.values())
plt.xlabel("设备类型")
plt.ylabel("出现次数")
plt.title("登录历史中不同设备使用比例")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("10G_data_new/device_usage_distribution.png")
plt.show()

end_time = time.time()
print(f"\n数据统计和处理耗时: {end_time - start_time:.2f} 秒")
