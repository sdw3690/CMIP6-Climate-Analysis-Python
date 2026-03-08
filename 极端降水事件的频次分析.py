import matplotlib
matplotlib.use('TkAgg')  # 设置为交互式后端
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import Normalize

# 读取降水数据
ds = xr.open_dataset("C:\\Users\\Zuim\\Desktop\\Datasource\\Gis Data\\CN05.1_Pre_1961_2018_daily_025x025.nc")

# 检查降水数据的单位
print(ds['pre'])  # 打印数据描述，检查单位是否为 mm/day

# 定义极端降水的阈值（95分位数）
threshold = ds['pre'].quantile(0.95, dim='time')  # 按时间维度计算95分位数

# 标记极端降水事件（1 表示极端降水事件，0 表示非极端降水事件）
extreme_events = ds['pre'] > threshold  # 得到布尔数组

# 按年份统计极端降水事件的频次
extreme_event_count = extreme_events.groupby('time.year').sum(dim='time')  # 每年每个网格的频次
extreme_event_mean = extreme_event_count.mean(dim='year')  # 多年平均频次

# 确保经纬度网格化
lon, lat = np.meshgrid(ds['lon'], ds['lat'])

# 添加阈值过滤，排除低降水区域（假设阈值低于 2 mm/day 的区域为无效）
extreme_event_mean = extreme_event_mean.where(threshold > 0, other=0)


# 动态调整色标范围
low_percentile = np.nanpercentile(extreme_event_mean.values, 1)  # 1%分位数
high_percentile = np.nanpercentile(extreme_event_mean.values, 99)  # 99%分位数

# 检查分位数是否有效
if np.isnan(low_percentile) or np.isnan(high_percentile) or low_percentile >= high_percentile:
    print("分位数计算出错或范围错误，使用默认值")
    low_percentile = 0.1  # 默认的最低值
    high_percentile = 10  # 默认的最高值

# 检查掩膜后数据是否全部为空
if extreme_event_mean.isnull().all():
    print("所有数据被掩膜，检查阈值条件")
    extreme_event_mean = extreme_event_mean.fillna(0)  # 替换 NaN 为 0

# 确保 levels 是递增的
levels = np.linspace(low_percentile, high_percentile, num=10)
print(f"最终使用的色标范围: {levels}")

# 改用线性色标
norm = Normalize(vmin=low_percentile, vmax=high_percentile)

# 设置地图投影
proj = ccrs.PlateCarree()

# 绘图
fig = plt.figure(figsize=(21, 13), dpi=500)
ax = fig.add_subplot(1, 1, 1, projection=proj)

# 设置地图范围（扩展至覆盖九段线区域）
region = [70, 140, 4, 55]
ax.set_extent(region, crs=proj)
ax.set_title('Annual Extreme Precipitation Events Frequency (Filtered)', loc='left', fontsize=10)

# 绘制极端降水事件频次分布图（线性色标）
contours = ax.contourf(lon, lat, extreme_event_mean, levels=levels, cmap=plt.get_cmap("Reds"), norm=norm)

# 添加色标
cbar = fig.colorbar(contours, shrink=0.5, orientation='horizontal', pad=0.05)
cbar.set_label('Extreme Event Frequency', fontsize=10)

# 绘制省界
china_shp = 'C:\\Users\\Zuim\\Desktop\\Datasource\\大创制图\\区县级行政区划数据-审图号：GS（2022）1873号\\ys\\省.shp'
china = shpreader.Reader(china_shp).geometries()
ax.add_geometries(china, proj, facecolor='none', edgecolor='black', zorder=1)

# 添加中国九段线
nine_dotted_line_shp = 'C:\\Users\\Zuim\\Desktop\\Datasource\\大创制图\\区县级行政区划数据-审图号：GS（2022）1873号\\ys\\线.shp'
nine_dotted_line = shpreader.Reader(nine_dotted_line_shp).geometries()
ax.add_geometries(
    nine_dotted_line, proj, facecolor='none', edgecolor='black', linestyle='--', linewidth=0.8, zorder=2
)

# 网格设置
gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7)
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 8, "color": 'k'}
gl.ylabel_style = {'size': 8, 'color': 'k'}
gl.top_labels = False
gl.right_labels = False

# 保存图片
save_path = r'C:\Users\Zuim\Desktop\Datasource\Gis Data\Output png\extreme_event_frequency_filtered_linear.png'
plt.savefig(save_path, bbox_inches='tight')
print(f"极端降水事件频次图已保存至 {save_path}")
