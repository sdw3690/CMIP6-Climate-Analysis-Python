# 差值地图重点区域红框.py
# UKESM1-0-LL zooc difference map (SSP585 - SSP119) + highlighted regions (red boxes)
# 兼容 ORCA 曲线网格 (j,i) + 2D 经纬度 (latitude/longitude 或 nav_lat/nav_lon)
# ✅ 自动识别时间坐标/时间维度（time / time_counter / t / ...）
# ✅ 360_day 安全年份筛选（按年份，不用日期字符串）
# ✅ 稳健色标（避免 vmax=nan/0 导致“没颜色”）
# ✅ 红框 + 箭头标注
# 输出：map_diff_ssp585-ssp119_<FIELD>_<YYYY-YYYY>_annotated.png

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ================== 路径（按需修改） ==================
DIR_LOW  = Path(r"F:\1A_Files\Zooc\ssp119")
DIR_HIGH = Path(r"F:\1A_Files\Zooc\ssp585")
OUTDIR   = Path(r"F:\1A_Files\Zooc\figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ================== 参数 ==================
FIELD = "surface"          # "surface" 或 "0-200m"
FUTURE0, FUTURE1 = 2051, 2100

# 研究区域（名称, lon_min, lon_max, lat_min, lat_max）
# ⚠️ 这里默认 0–360 经度体系（UKESM 常见 nav_lon/longitude 为 0–360）
REGIONS = [
    ("South China Sea",        105, 120,   0,  25),
    ("Arabian Sea",             50,  75,   5,  25),
    ("N. Atlantic (high lat)", 300, 350,  45,  65),  # -60–-10 → 300–350
    ("Southern Ocean",           0, 360, -60, -40),
    ("Equatorial Pacific",     190, 240,  -5,   5),  # -170–-120 → 190–240
]

# ================== 工具函数 ==================
def pick_one_nc(folder: Path) -> Path:
    ncs = sorted(folder.glob("*.nc"))
    if not ncs:
        raise FileNotFoundError(f"No .nc found in: {folder}")
    return ncs[0]

def open_ds(folder: Path) -> xr.Dataset:
    """
    ✅ 读取文件夹下所有 nc，按 time 拼接（open_mfdataset）
    ✅ chunks={"time":12} 分块，避免爆内存（若时间维不叫 time，会自动再按真实时间维 rechunk）
    ✅ 自动 sortby("time") + 去重（防止文件边界重叠）
    ✅ 仍兼容 2D 经纬度（curvilinear）和 360_day（use_cftime=True）
    """
    ncs = sorted(folder.glob("*.nc"))
    if not ncs:
        raise FileNotFoundError(f"No .nc found in: {folder}")

    # 先按要求给 time 分块（如果真实时间维不是 time，下面会自动再按真实维度 chunk）
    ds = xr.open_mfdataset(
        [str(p) for p in ncs],
        combine="by_coords",
        use_cftime=True,
        chunks={"time": 12},
        parallel=False,
        coords="minimal",
        data_vars="minimal",
        compat="override",
    )

    # --- 自动识别 ds 里的时间坐标/维度，并 sort + 去重 ---
    def _find_time_coord_and_dim_in_ds(_ds: xr.Dataset):
        # 常见名字优先
        for c in ["time", "time_counter", "t", "time_centered", "time_bounds"]:
            if c in _ds.coords and hasattr(_ds[c], "dt"):
                return c, _ds[c].dims[0]
        # 兜底：找任何支持 .dt 的坐标
        for c in _ds.coords:
            if hasattr(_ds[c], "dt"):
                return c, _ds[c].dims[0]
        raise KeyError(f"No datetime-like time coordinate found in dataset. coords={list(_ds.coords)}")

    tcoord, tdim = _find_time_coord_and_dim_in_ds(ds)

    # sortby time
    ds = ds.sortby(tcoord)

    # 去重（处理文件边界重叠导致的重复时间点）
    # 用 pandas.Index 的 duplicated 对 cftime/object 更稳健
    import pandas as pd
    idx = pd.Index(ds[tcoord].values)
    keep = ~idx.duplicated(keep="first")
    if keep.sum() != keep.size:
        ds = ds.isel({tdim: np.where(keep)[0]})

    # 如果真实时间维不叫 "time"，再确保按真实时间维进行 chunk
    # （满足“避免爆内存”的目标；同时保留你要求的 chunks={"time":12} 入参）
    if tdim != "time":
        ds = ds.chunk({tdim: 12})
    else:
        ds = ds.chunk({"time": 12})

    return ds

def find_time_coord_and_dim(da: xr.DataArray):
    """
    返回 (time_coord_name, time_dim_name)
    兼容 time / time_counter / t / time_centered 等。
    """
    # 优先常见名字
    for c in ["time", "time_counter", "t", "time_centered", "time_bounds"]:
        if c in da.coords and hasattr(da[c], "dt"):
            return c, da[c].dims[0]

    # 兜底：找任何支持 .dt 的坐标
    for c in da.coords:
        if hasattr(da[c], "dt"):
            return c, da[c].dims[0]

    raise KeyError(f"No datetime-like time coordinate found. coords={list(da.coords)}")

def sel_years(da: xr.DataArray, start: int, end: int) -> xr.DataArray:
    """360_day 安全年份筛选：按年份布尔索引，不用日期字符串。"""
    tcoord, tdim = find_time_coord_and_dim(da)
    yrs = da[tcoord].dt.year
    return da.isel({tdim: (yrs >= start) & (yrs <= end)})

def select_surface_or_0_200m(da: xr.DataArray, field: str) -> xr.DataArray:
    depth_candidates = ["lev", "depth", "olevel", "deptht", "z_t"]
    dname = next((d for d in depth_candidates if d in da.dims), None)
    if dname is None:
        return da

    if field == "surface":
        return da.isel({dname: 0})

    if field == "0-200m":
        try:
            sub = da.sel({dname: slice(0, 200)})
            if sub.sizes.get(dname, 0) >= 1:
                return sub.mean(dname, skipna=True)
        except Exception:
            pass
        return da.isel({dname: slice(0, 5)}).mean(dname, skipna=True)

    raise ValueError("FIELD must be 'surface' or '0-200m'")

def robust_symmetric_limits(arr, pct=98):
    """稳健计算对称色标范围，避免 vmax=nan/0 导致无颜色。"""
    vals = np.asarray(arr)
    finite = np.isfinite(vals)
    if finite.sum() == 0:
        raise ValueError("All-NaN map. Check time selection / masking / variable.")
    vmax = np.nanpercentile(np.abs(vals[finite]), pct)
    if (not np.isfinite(vmax)) or vmax == 0:
        vmax = np.nanmax(np.abs(vals[finite]))
    if vmax == 0:
        vmax = 1e-12
    return -vmax, vmax

def get_lat_lon_2d(ds: xr.Dataset):
    """
    获取 2D 纬度/经度变量：
    常见：latitude/longitude 或 nav_lat/nav_lon 或 lat/lon（2D）
    """
    lat_candidates = ["latitude", "nav_lat", "lat"]
    lon_candidates = ["longitude", "nav_lon", "lon"]

    lat_name = next((n for n in lat_candidates if n in ds.variables), None)
    lon_name = next((n for n in lon_candidates if n in ds.variables), None)

    if lat_name is None or lon_name is None:
        raise KeyError(f"Cannot find 2D lat/lon in dataset. variables={list(ds.variables)[:50]} ...")

    return ds[lat_name], ds[lon_name], lat_name, lon_name

# ================== 主流程 ==================
ds_low  = open_ds(DIR_LOW)
ds_high = open_ds(DIR_HIGH)

# 变量名兜底
var = "zooc" if "zooc" in ds_low.data_vars else list(ds_low.data_vars.keys())[0]
da_low  = ds_low[var]
da_high = ds_high[var]
units = da_low.attrs.get("units", "")

# 2D 经纬度（ORCA 曲线网格）
lat2d, lon2d, lat_name, lon_name = get_lat_lon_2d(ds_low)

# 海洋掩膜：用低排放的一个时刻一层作为 mask（finite 视为海洋）
# depth 维名可能不是 lev，这里稳健找一下
depth_candidates = ["lev", "depth", "olevel", "deptht", "z_t"]
dname = next((d for d in depth_candidates if d in da_low.dims), None)
if dname is None:
    mask_ocean = np.isfinite(da_low.isel({find_time_coord_and_dim(da_low)[1]: 0}))
else:
    tdim0 = find_time_coord_and_dim(da_low)[1]
    mask_ocean = np.isfinite(da_low.isel({tdim0: 0, dname: 0}))

# 选 surface / 0–200m
da_low2  = select_surface_or_0_200m(da_low,  FIELD)
da_high2 = select_surface_or_0_200m(da_high, FIELD)

# 自动找时间维度名（低/高可能一致，但也可能不同）
tcoord_l, tdim_l = find_time_coord_and_dim(da_low2)
tcoord_h, tdim_h = find_time_coord_and_dim(da_high2)

print("Debug:",
      f"var={var}",
      f"low dims={da_low2.dims}",
      f"high dims={da_high2.dims}",
      f"time(low)={tcoord_l}/{tdim_l}",
      f"time(high)={tcoord_h}/{tdim_h}",
      f"lat2d={lat_name} dims={lat2d.dims}",
      f"lon2d={lon_name} dims={lon2d.dims}",
      sep="\n  ")

# 未来时期平均（按年份筛选，兼容 360_day）+ 海洋 mask
lo_fut = sel_years(da_low2,  FUTURE0, FUTURE1).mean(tdim_l, skipna=True).where(mask_ocean)
hi_fut = sel_years(da_high2, FUTURE0, FUTURE1).mean(tdim_h, skipna=True).where(mask_ocean)

# 差值：SSP585 - SSP119
diff = (hi_fut - lo_fut).where(mask_ocean)

# ====== 绘图 ======
proj = ccrs.Robinson()
pc = ccrs.PlateCarree()

fig = plt.figure(figsize=(13, 5))
ax = plt.axes(projection=proj)
ax.set_global()

ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)

vmin, vmax = robust_symmetric_limits(diff.values, pct=98)

im = ax.pcolormesh(
    lon2d, lat2d, diff,
    transform=pc, shading="auto",
    cmap="RdBu_r", vmin=vmin, vmax=vmax, zorder=1
)

cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.9)
cb.set_label(f"zooc difference (SSP585 − SSP119), {FIELD}, {FUTURE0}–{FUTURE1}  {units}")

ax.set_title("Zooplankton carbon difference (SSP585 − SSP119) with key regions highlighted")

# 红框 + 标注
# 红框（不加箭头和文字，后续手动标注）
for _, lon0, lon1, lat0, lat1 in REGIONS:
    xs = [lon0, lon1, lon1, lon0, lon0]
    ys = [lat0, lat0, lat1, lat1, lat0]
    ax.plot(
        xs, ys,
        transform=pc,
        linewidth=2.2,
        color="red",
        zorder=4
    )


out = OUTDIR / f"map_diff_ssp585-ssp119_{FIELD}_{FUTURE0}-{FUTURE1}_annotated.png"
plt.tight_layout()
plt.savefig(out, dpi=300)
plt.show()

print("✅ Saved:", out)
print("Debug:",
      "diff finite ratio =", float(np.isfinite(diff.values).mean()),
      "min/max =", float(np.nanmin(diff.values)), float(np.nanmax(diff.values)),
      "vmin/vmax =", float(vmin), float(vmax))
