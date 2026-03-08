import matplotlib
matplotlib.use('TkAgg')  # 设置 Matplotlib 的后端为 TkAgg，适合交互式环境
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader  # 用于读取 Shapefile 数据
import cartopy.crs as ccrs  # 用于地图投影
import numpy as np
import xarray as xr  # 用于处理 NetCDF 数据
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 用于格式化经纬度标签

# Step 1: 数据读取与检查
# 读取 NetCDF 数据文件
print("正在读取降水数据...")
dataset_path = "C:\\Users\\Zuim\\Desktop\\Datasource\\Gis Data\\CN05.1_Pre_1961_2018_daily_025x025.nc"
ds = xr.open_dataset(dataset_path)
print("数据读取完成！")

# 检查数据的单位及基本信息
print("降水数据的信息：")
print(ds['pre'])  # 打印数据元信息，确保其单位为 mm/day 或其他合理的形式

# Step 2: 数据处理与降水量计算
# 计算年总降水量（如果单位为 mm/day，则总量需要乘以天数）
print("正在计算每年的总降水量...")
rain = ds['pre'].groupby('time.year').sum(dim='time')  # 按年份分组，求和计算年总降水量
rain_mean = rain.mean(dim='year')  # 多年平均降水量
print("多年平均降水量计算完成！")

# 网格化经纬度，确保数据能用于绘图
print("正在创建经纬度网格...")
lon, lat = np.meshgrid(ds['lon'], ds['lat'])

# 动态计算降水数据范围，确保色标范围合理
print("正在分析降水数据范围...")
rain_min = np.nanmin(rain_mean.values)  # 找到降水量的最小值（忽略 NaN）
rain_max = np.nanmax(rain_mean.values)  # 找到降水量的最大值（忽略 NaN）
print(f"降水量范围：最小值 = {rain_min:.2f} mm，最大值 = {rain_max:.2f} mm")

# 为绘图设置色标的范围，分为 10 个等级
levels = np.linspace(rain_min, rain_max, num=10)

# Step 3: 绘图与可视化
# 设置地图投影为 PlateCarree
proj = ccrs.PlateCarree()

# 创建绘图画布，设置画布大小和分辨率
print("正在初始化绘图...")
fig = plt.figure(figsize=(21, 13), dpi=500)
ax = fig.add_subplot(1, 1, 1, projection=proj)

# 设置地图范围（包括中国及南海九段线区域）
region = [70, 140, 4, 55]  # 经度范围 [70, 140]，纬度范围 [4, 55]
ax.set_extent(region, crs=proj)

# 设置标题，字体大小为 10
ax.set_title('Annual Extreme Precipitation (mm)', loc='left', fontsize=10)

# 绘制平均年降水量的填色图
print("正在绘制平均降水量分布图...")
contours = ax.contourf(
    lon, lat, rain_mean, levels=levels, extend='both', cmap=plt.get_cmap("Blues")
)

# 添加色标并设置标签
print("正在添加色标...")
cbar = fig.colorbar(contours, shrink=0.5, orientation='horizontal', pad=0.05)
cbar.set_label('Extreme Precipitation (mm)', fontsize=10)

# 添加中国省界
print("正在绘制中国省界...")
china_shp = 'C:\\Users\\Zuim\\Desktop\\Datasource\\大创制图\\区县级行政区划数据-审图号：GS（2022）1873号\\ys\\省.shp'
china = shpreader.Reader(china_shp).geometries()
ax.add_geometries(china, proj, facecolor='none', edgecolor='black', zorder=1)

# 添加中国九段线
print("正在添加中国九段线...")
nine_dotted_line_shp = 'C:\\Users\\Zuim\\Desktop\\Datasource\\大创制图\\区县级行政区划数据-审图号：GS（2022）1873号\\ys\\线.shp'
nine_dotted_line = shpreader.Reader(nine_dotted_line_shp).geometries()
ax.add_geometries(
    nine_dotted_line, proj, facecolor='none', edgecolor='black', linestyle='--', linewidth=0.8, zorder=2
)

# 设置经纬度网格和格式
print("正在设置经纬度网格...")
gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7)
gl.xformatter = LongitudeFormatter()  # 格式化经度为标准格式
gl.yformatter = LatitudeFormatter()  # 格式化纬度为标准格式
gl.xlabel_style = {'size': 8, "color": 'k'}  # 设置经度标签样式
gl.ylabel_style = {'size': 8, 'color': 'k'}  # 设置纬度标签样式
gl.top_labels = False  # 不显示顶部标签
gl.right_labels = False  # 不显示右侧标签

# Step 4: 保存和展示
# 设置保存路径并保存图片
save_path = r'C:\Users\Zuim\Desktop\Datasource\Gis Data\Output png\output_with_nine_dotted_line_updated.png'
plt.savefig(save_path, bbox_inches='tight')  # 保存图片并裁剪多余空白
print(f"图表已保存至 {save_path}")

# 显示绘图
plt.show()
print("绘图完成！")
