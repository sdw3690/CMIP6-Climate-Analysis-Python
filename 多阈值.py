import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# Step 1: 数据读取与检查
dataset_path = "C:\\Users\\Zuim\\Desktop\\Datasource\\Gis Data\\CN05.1_Pre_1961_2018_daily_025x025.nc"
ds = xr.open_dataset(dataset_path)
print("数据读取完成！")
print(ds['pre'])

# Step 2: 定义多个阈值
thresholds = [10, 20, 50]  # 定义降水阈值（单位 mm/day）

# Step 3: 基于阈值的累计频次分析
results = {}
for threshold in thresholds:
    print(f"正在处理阈值 {threshold} mm...")
    extreme_events = ds['pre'] > threshold  # 筛选超过阈值的降水事件
    event_frequency = extreme_events.groupby('time.year').sum(dim='time')  # 按年统计频次
    results[f"{threshold} mm"] = event_frequency.mean(dim='year')  # 计算多年平均频次

# Step 4: 绘制每个阈值的空间分布
proj = ccrs.PlateCarree()
region = [70, 140, 4, 55]  # 地图范围
lon, lat = np.meshgrid(ds['lon'], ds['lat'])

for label, rain_mean in results.items():
    # 动态调整色标范围
    rain_min = np.nanpercentile(rain_mean.values, 5)  # 5% 分位数
    rain_max = np.nanpercentile(rain_mean.values, 95)  # 95% 分位数
    levels = np.linspace(rain_min, rain_max, num=10)

    # 创建绘图
    fig = plt.figure(figsize=(21, 13), dpi=500)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(region, crs=proj)
    ax.set_title(f'Annual Frequency Above {label}', loc='left', fontsize=10)

    # 绘制填色图
    contours = ax.contourf(
        lon, lat, rain_mean, levels=levels, extend='both', cmap=plt.get_cmap("Reds")
    )

    # 添加色标
    cbar = fig.colorbar(contours, shrink=0.5, orientation='horizontal', pad=0.05)
    cbar.set_label(f'Frequency Above {label}', fontsize=10)

    # 绘制省界
    china_shp = 'C:\\Users\\Zuim\\Desktop\\Datasource\\大创制图\\区县级行政区划数据-审图号：GS（2022）1873号\\ys\\省.shp'
    china = shpreader.Reader(china_shp).geometries()
    ax.add_geometries(china, proj, facecolor='none', edgecolor='black', zorder=1)

    # 添加九段线
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

    # 保存图片路径
    save_path = f'C:\\Users\\Zuim\\Desktop\\Datasource\\Gis Data\\Output png\\output_frequency_threshold_{label.replace(" ", "_")}.png'
    plt.savefig(save_path, bbox_inches='tight')  # 保存图片并裁剪多余空白
    print(f"图表已保存至 {save_path}")

    plt.show()
