import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

# 读取降水数据
ds = xr.open_dataset("C:\\Users\\Zuim\\Desktop\\Datasource\\Gis Data\\CN05.1_Pre_1961_2018_daily_025x025.nc")

# 定义极端降水的阈值（95分位数）
threshold = ds['pre'].quantile(0.95, dim='time')  # 按时间维度计算95分位数

# 标记极端降水事件（1 表示极端降水事件，0 表示非极端降水事件）
extreme_events = ds['pre'] > threshold  # 得到布尔数组

# 添加年份作为一个新的维度
extreme_events['year'] = extreme_events['time.year']  # 提取年份并添加为数据变量

# 按年份统计极端降水事件频次总和（每年的事件总数）
extreme_event_count = extreme_events.groupby('year').sum(dim=['lat', 'lon', 'time'])  # 按空间和时间维度求和

# 打印分组后的数据结构
print(extreme_event_count)

# 提取年份和频次
years = extreme_event_count['year'].values  # 提取年份
frequencies = extreme_event_count.values  # 提取频次数据

# 检查 NaN 问题
if np.isnan(frequencies).all():
    print("所有频次数据为 NaN，请检查数据完整性。")
    frequencies = [0] * len(years)

# 绘制按年份统计的趋势线
plt.figure(figsize=(10, 6))
plt.plot(years, frequencies, marker='o', label='Extreme Events Frequency')
plt.title('Trend of Extreme Precipitation Events Frequency')
plt.xlabel('Year')
plt.ylabel('Total Frequency')
plt.grid(True)
plt.legend()
plt.show()
