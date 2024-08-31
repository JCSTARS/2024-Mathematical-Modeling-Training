import pandas as pd
df = pd.read_excel('2009B/Attachment.xlsx', engine='openpyxl')
df = df.head(350)
bai_type = df[(df['类型'] == '白内障')|(df['类型'] == '白内障(双眼)')]
wai_type = df[(df['类型'] == '外伤')]
else_type = df[(df['类型'] == '青光眼')|(df['类型'] == '视网膜疾病')]
print("白内障病人：", bai_type.shape[0])
print("外伤病人：", wai_type.shape[0])
print("其他病人：", else_type.shape[0])

# 转换“门诊时间”列为日期格式
df['门诊时间'] = pd.to_datetime(df['门诊时间'])

# 设置日期范围（从2008-7-13到2008-9-4）
start_date = '2008-7-13'
end_date = '2008-9-4'

# 创建日期范围
date_range = pd.date_range(start=start_date, end=end_date)

results_df = pd.DataFrame(columns=['日期', '白内障病人数', '外伤病人数', '其他病人数'])
# 遍历日期范围，并筛选每一天的记录
for single_date in date_range:
    print(f"Processing data for: {single_date.strftime('%Y-%m-%d')}")
    daily_patients = df[df['门诊时间'] == single_date.strftime('%Y-%m-%d')]
    bai_daily = daily_patients[(daily_patients['类型'] == '白内障')|(daily_patients['类型'] == '白内障(双眼)')]
    wai_daily = daily_patients[(daily_patients['类型'] == '外伤')]
    else_daily = daily_patients[(daily_patients['类型'] == '青光眼')|(daily_patients['类型'] == '视网膜疾病')]
    
    results_df = results_df.append({'日期': single_date, '白内障病人数': bai_daily.shape[0], '外伤病人数': wai_daily.shape[0], '其他病人数': else_daily.shape[0]}, ignore_index=True)
results_df.to_excel('2009B/daily_patient_counts.xlsx')