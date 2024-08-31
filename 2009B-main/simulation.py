import pandas as pd

from datetime import timedelta

df = pd.read_excel('2009B/Attachment.xlsx', engine='openpyxl')
df = df.tail(183)
df = df.head(51)
# 处理为日期格式
for column in ['门诊时间', '入院时间', '第一次手术时间', '第二次手术时间', '出院时间']:
    df[column] = pd.to_datetime(df[column], errors='coerce')  
    # 使用errors='coerce'来处理无效的日期格式

replace_dict = {'青光眼': '其他', '视网膜疾病': '其他'}
df['类型'] = df['类型'].replace(replace_dict)
df = df.sort_values(by=['类型','门诊时间'])
df.to_excel('2009B/group.xlsx')
df = df.tail(8)


# 模拟病人流程
def simulate_patient_flow(df):
    # 设置平均住院时间和其他参数
    average_stay_other = timedelta(days=10)
    beds_available = 29

# 初始化变量
    current_admissions = 0
    admission_queue = []  # 存储等待入院的病人信息
    for index, row in df.iterrows():
        # 计算出院时间
        discharge_date = row['门诊时间'] + average_stay_other
    
        # 如果有空床，则入院
        if current_admissions < beds_available:
            row['入院时间'] = row['门诊时间']
            df.at[index, '入院时间'] = row['门诊时间']
            current_admissions += 1
        else:
        # 否则，加入等待队列
            admission_queue.append((index, row['门诊时间'], discharge_date))
    
        # 检查是否有病人可以出院
        while admission_queue and (df.loc[admission_queue[0][0], '门诊时间'] + average_stay_other <= discharge_date):
            _, wait_time, _ = admission_queue.pop(0)
            current_admissions -= 1
            df.at[current_admissions, '入院时间'] = wait_time
            # df.at[current_admissions, '入院时间'] = wait_time

        # 更新出院时间
        row['出院时间'] = discharge_date
        df.at[index, '出院时间'] = row['门诊时间']

    return df

# 调用函数
df_result = simulate_patient_flow(df)

# 保存结果
df_result.to_excel('patient_flow_simulation.xlsx', index=False)

"""
days_to_subtract = {
    '白内障(双眼)': -2,
    '白内障': -2,
    '外伤': -1,
    '其他': -3  # 包括青光眼和视网膜病人
}

# 平均等待时间=第一次手术时间-门诊时间-有效术前时间
df['等待时间'] = df.apply(
    lambda row: (row['第一次手术时间'] - row['门诊时间']).days + days_to_subtract.get(row['类型'], 0),
    axis=1
)

average_wait_times = df.groupby('类型')['等待时间'].mean()
print(average_wait_times)
"""