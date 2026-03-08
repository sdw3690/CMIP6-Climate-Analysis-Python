import matplotlib
matplotlib.use('TkAgg')  # 设置为交互式后端
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm
import matplotlib.ticker
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator

# 地图绘制和投影库
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# 数据处理与分析库
import numpy as np
import math
import xarray as xr
import cftime as datetime
from sklearn.linear_model import LinearRegression
import pymannkendall as mk
import dask  # 引入dask库

# 修改为新的数据路径
ds1 = xr.open_dataset("C:\\Users\\Zuim\\Desktop\\Datasource\\Gis Data\\CN05.1_Pre_1961_2018_daily_025x025.nc")
print(ds1)

# 使用chunk分块处理单一数据集，避免内存问题
ds_all = ds1.chunk({'time': 1000})

# 将处理后的数据保存为新的 NetCDF 文件，使用 netCDF4 后端
output_path = "C:\\Users\\Zuim\\Desktop\\Datasource\\Gis Data\\rainsp.nc"
ds_all.to_netcdf(output_path, engine="netcdf4")

# 读取并处理拼接后的文件（实际上这里已经只处理一个文件）
ds = xr.open_dataset(output_path)
ds = xr.where(ds >= 50, 1, 0)  # 计算极端降水频次

lon = ds['lon']
lat = ds['lat']
mean_mon = ds.groupby('time.month').mean('time')  # 按月计算均值
mean_year = ds.mean('time')  # 求年平均
year_all = mean_year * 365
rain = year_all['pre']

# 绘图部分（保持不变）
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(21, 13), dpi=500)  # 设置图片大小和分辨率
ax = fig.add_subplot(1, 1, 1, projection=proj)
region = [70, 136, 15, 55]
ax.set_extent(region, crs=proj)
ax.set_title('Annual Extreme Precipitation(mm)', loc='left', fontsize=10)

gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7)
x_major_locator = MultipleLocator(10)
y_major_locator = MultipleLocator(10)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.ylim(15, 55)
plt.xlim(70, 140)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 8, "color": 'k'}
gl.ylabel_style = {'size': 8, 'color': 'k'}
gl.top_labels = False
gl.right_labels = False

# 颜色映射
levels = [0, 10, 25, 50, 100, 150, 175]
cmap = plt.get_cmap("Blues")
contours = ax.contourf(lon, lat, rain, levels=levels, extend='both', cmap=cmap)

# 检查并设置动态 levels
rain_min = np.nanmin(rain.values)
rain_max = np.nanmax(rain.values)
if rain_min == rain_max:
    rain_max += 1  # 避免 min 和 max 相同导致 levels 无法生成
levels = np.linspace(rain_min, rain_max, num=10)  # 动态生成等间隔 levels

# 绘制等值线图
contours = ax.contourf(lon, lat, rain, levels=levels, extend='both', cmap=plt.get_cmap("Blues"))

# 添加色标
cbar = fig.colorbar(contours, shrink=0.5, orientation='horizontal')
cbar.set_label('Extreme Precipitation (mm)', fontsize=10)


# 绘制省级行政区边界和南海子图
china = shpreader.Reader('C:\\Users\\Zuim\\Desktop\\Datasource\\大创制图\\区县级行政区划数据-审图号：GS（2022）1873号\\ys\\省.shp').geometries()
pro = shpreader.Reader('C:\\Users\\Zuim\\Desktop\\Datasource\\大创制图\\区县级行政区划数据-审图号：GS（2022）1873号\\ys\\线.shp').geometries()

# 添加主图中的省界
ax.add_geometries(china, proj, facecolor='none', edgecolor='black', zorder=1)

# 添加南海子图
sub_ax = fig.add_axes([0.75, 0.15, 0.2, 0.2], projection=proj)
sub_ax.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
sub_ax.add_geometries(pro, ccrs.PlateCarree(), edgecolor='black', zorder=1)
sub_ax.contourf(lon, lat, rain, levels=levels, extend='both', cmap=cmap, zorder=0)

# 保存图片
save_path = r'C:\Users\Zuim\Desktop\Datasource\Gis Data\Output png\output.png'
plt.savefig(save_path)
print(f"图表已保存至 {save_path}")

# 纬度加权平均
EARTH_RADIUS = 6371000  # 地球半径，单位为米
DEG2RAD = math.pi / 180  # 角度转弧度的系数

# 计算每个格点的面积
def calc_grid_area(lat, lon):
    dlon = np.diff(lon) * DEG2RAD
    dlat = np.diff(lat) * DEG2RAD

    # 使用完整的纬度网格
    dlat_grid, dlon_grid = np.meshgrid(dlat, dlon, indexing='ij')
    lat_grid, _ = np.meshgrid(lat[:-1], lon[:-1], indexing='ij')

    dA = EARTH_RADIUS ** 2 * dlat_grid * dlon_grid * np.cos(lat_grid * DEG2RAD)
    return dA

# 读取 NetCDF 文件
file_path = 'C:\\Users\\Zuim\\Desktop\\Datasource\\Gis Data\\CN05.1_Pre_1961_2018_daily_025x025.nc'
ds = xr.open_dataset(file_path)

lat = ds['lat'].values
lon = ds['lon'].values
precip = ds['pre']

dA = calc_grid_area(lat, lon)
weights = xr.DataArray(dA, dims=["lat", "lon"], coords={"lat": lat[:-1], "lon": lon[:-1]})

# 定义阈值
thresholds = [10, 20, 50]
extreme_precip_days = {}

for threshold in thresholds:
    extreme_days = (precip >= threshold) * 1
    extreme_days_wgt = extreme_days.isel(lat=slice(0, -1), lon=slice(0, -1)) * weights
    extreme_days_areaave = extreme_days_wgt.sum(dim=("lon", "lat")) / weights.sum()
    extreme_precip_days[threshold] = extreme_days_areaave.groupby('time.year').sum(dim='time')

# 可视化极端降水天数随年份的变化
fig, ax = plt.subplots(figsize=(10, 6))
for threshold, result in extreme_precip_days.items():
    ax.plot(result['year'], result, label=f'Threshold: {threshold} mm')

ax.set_title('Annual Extreme Precipitation Days in China (Multiple Thresholds)')
ax.set_xlabel('Year')
ax.set_ylabel('Extreme Precipitation Days')
ax.legend()
plt.grid(True)
plt.show()
