import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

print("\n=== Figure A: 5 regions × 2 models × 2 scenarios (SSP119 vs SSP585) ===\n")

# ---------------- 输出路径 ----------------
OUTDIR = Path(r"F:\1A_Files\Zooc\figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------- 参数 ----------------
FIELD = "surface"
YEAR0, YEAR1 = 1950, 2100
T_START = f"{YEAR0}-01-01"
T_END   = f"{YEAR1}-12-30"   # 360_day 安全

# ---------------- 区域：5个 ----------------
# 注意 tuple: (name, lon0, lon1, lat0, lat1)
REGIONS = [
    ("North Atlantic (high lat)", -60,  -10,  45,  65),
    ("Arabian Sea",                50,   75,   5,  25),
    ("South China Sea",           105,  120,   0,  25),
    ("Southern Ocean",              0,  360, -60, -40),
    ("Equatorial Pacific (Nino3.4)",190, 240,  -5,   5),
]

# ---------------- 模型路径 ----------------
# UKESM：沿用你原来的两个scenario文件夹（多数情况下已包含 historical+scenario）
DIR_UK_SSP119 = Path(r"F:\1A_Files\Zooc\ssp119")
DIR_UK_SSP585 = Path(r"F:\1A_Files\Zooc\ssp585")

# IPSL：你截图里 historical 在这里（两段文件）
DIR_IPSL_HIST  = Path(r"F:\1A_Files\Zooc_IPSL-CM6A-LR\historic")

# ⚠️ 下面两行请确认你的未来情景文件夹名是否真是 ssp119 / ssp585
DIR_IPSL_SSP119 = Path(r"F:\1A_Files\Zooc_IPSL-CM6A-LR\ssp119")
DIR_IPSL_SSP585 = Path(r"F:\1A_Files\Zooc_IPSL-CM6A-LR\ssp585")

MODELS = {
    "UKESM1-0-LL": {
        "historical": None,            # 如果你UKESM历史在单独文件夹，可填Path；否则留None
        "ssp119": DIR_UK_SSP119,
        "ssp585": DIR_UK_SSP585,
        "linestyle": "-",              # 主模型：实线
    },
    "IPSL-CM6A-LR": {
        "historical": DIR_IPSL_HIST,   # IPSL历史单独文件夹
        "ssp119": DIR_IPSL_SSP119,
        "ssp585": DIR_IPSL_SSP585,
        "linestyle": "--",             # 对比模型：虚线
    }
}

SCENARIOS = {
    "ssp119": {"label": "SSP1-1.9", "color": "C0"},
    "ssp585": {"label": "SSP5-8.5", "color": "C1"},
}

# ---------------- 工具函数 ----------------
def open_da_mf(folder: Path) -> xr.DataArray:
    files = sorted(folder.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc found in: {folder}")

    ds = xr.open_mfdataset(
        [str(f) for f in files],
        combine="by_coords",
        use_cftime=True,
        chunks={"time": 12},
        parallel=False
    )

    var = "zooc" if "zooc" in ds.data_vars else list(ds.data_vars.keys())[0]
    da = ds[var]

    # 时间排序 + 去重
    if "time" in da.dims:
        da = da.sortby("time")
        t = da["time"].values
        _, idx = np.unique(t, return_index=True)
        da = da.isel(time=idx)

    return da

def select_field(da: xr.DataArray, field: str) -> xr.DataArray:
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

def find_latlon_vars(da: xr.DataArray):
    candidates_lat = ["lat", "latitude", "nav_lat", "TLAT", "yt_ocean"]
    candidates_lon = ["lon", "longitude", "nav_lon", "TLONG", "xt_ocean"]

    lat = None
    lon = None

    for k in candidates_lat:
        if k in da.coords:
            lat = da.coords[k]
            break
    for k in candidates_lon:
        if k in da.coords:
            lon = da.coords[k]
            break

    if lat is None or lon is None:
        ds = da._to_temp_dataset()
        if lat is None:
            for k in candidates_lat:
                if k in ds.variables:
                    lat = ds[k]
                    break
        if lon is None:
            for k in candidates_lon:
                if k in ds.variables:
                    lon = ds[k]
                    break

    if lat is None or lon is None:
        raise KeyError("找不到经纬度变量（lat/lon 或 nav_lat/nav_lon 等）。")

    return lat, lon

def guess_space_dims(da: xr.DataArray):
    non_space = {"time", "lev", "depth", "olevel", "deptht", "z_t"}
    space_dims = [d for d in da.dims if d not in non_space]
    if not space_dims:
        raise ValueError("无法识别空间维度。")
    return space_dims

def lon360(lon):
    return (lon + 360) % 360

def mask_region_0360(da: xr.DataArray, lat: xr.DataArray, lon: xr.DataArray,
                     lon0, lon1, lat0, lat1):
    """
    更稳健：无论数据lon是[-180,180]还是[0,360]，都转成0–360做掩膜
    支持跨日期变更（例如 lon0=350, lon1=20）
    """
    lon_d = lon360(lon)
    lon0_ = lon0 % 360
    lon1_ = lon1 % 360

    if lon0_ <= lon1_:
        lon_cond = (lon_d >= lon0_) & (lon_d <= lon1_)
    else:
        lon_cond = (lon_d >= lon0_) | (lon_d <= lon1_)

    lat_cond = (lat >= lat0) & (lat <= lat1)
    cond = lon_cond & lat_cond
    return da.where(cond)

def area_weighted_mean_masked(da_masked: xr.DataArray, lat: xr.DataArray):
    space_dims = guess_space_dims(da_masked)
    w = np.cos(np.deg2rad(lat))
    return da_masked.weighted(w).mean(dim=space_dims, skipna=True)

def annual_mean(ts):
    return ts.resample(time="YS").mean()

def linfit_r2(years, y):
    years = np.asarray(years, dtype=float)
    y = np.asarray(y, dtype=float)
    msk = np.isfinite(years) & np.isfinite(y)
    if msk.sum() < 2:
        return np.nan, np.nan, np.nan
    years = years[msk]
    y = y[msk]
    m, b = np.polyfit(years, y, 1)
    yhat = m * years + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return m, b, r2

def load_model_scenario(model_cfg, scen_key, year0_for_autoattach=1950):
    """
    读取某模型某情景：
    - 如果 model_cfg['historical'] 不是 None 且情景文件起始年份 > year0_for_autoattach，
      自动把 historical 拼到情景前面（适合 IPSL 的 split 布局）
    """
    da_scen = open_da_mf(model_cfg[scen_key])

    if model_cfg.get("historical") is not None:
        try:
            ymin = int(da_scen["time"].dt.year.min().compute())
        except Exception:
            ymin = 9999
        if ymin > year0_for_autoattach:
            da_hist = open_da_mf(model_cfg["historical"])
            da_scen = xr.concat([da_hist, da_scen], dim="time")

            # 再去重一次（防边界重叠）
            if "time" in da_scen.dims:
                da_scen = da_scen.sortby("time")
                t = da_scen["time"].values
                _, idx = np.unique(t, return_index=True)
                da_scen = da_scen.isel(time=idx)

    return da_scen

# ---------------- 预读取：2模型×2情景 ----------------
DATA = {}
for mname, mcfg in MODELS.items():
    DATA[mname] = {}
    for skey in ["ssp119", "ssp585"]:
        da = load_model_scenario(mcfg, skey, year0_for_autoattach=YEAR0)
        da = select_field(da, FIELD).sel(time=slice(T_START, T_END))
        lat, lon = find_latlon_vars(da)

        ymin = int(da["time"].dt.year.min().compute())
        ymax = int(da["time"].dt.year.max().compute())
        print(f"{mname} {skey} years: {ymin}–{ymax}")

        DATA[mname][skey] = {"da": da, "lat": lat, "lon": lon}

print()

# ---------------- 画图：5个区域，每个区域一个子图 ----------------
n = len(REGIONS)
fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, 3.2 * n), sharex=True)
if n == 1:
    axes = [axes]

for i, (ax, (rname, lon0, lon1, lat0, lat1)) in enumerate(zip(axes, REGIONS)):
    txt_lines = []

    # 为了避免legend重复，仅第一幅子图设置label
    set_label = (i == 0)

    for mname, mcfg in MODELS.items():
        ls = mcfg["linestyle"]

        # 取两情景数据
        da_119 = DATA[mname]["ssp119"]["da"]
        lat_119 = DATA[mname]["ssp119"]["lat"]
        lon_119 = DATA[mname]["ssp119"]["lon"]

        da_585 = DATA[mname]["ssp585"]["da"]
        lat_585 = DATA[mname]["ssp585"]["lat"]
        lon_585 = DATA[mname]["ssp585"]["lon"]

        # 掩膜+区域平均+年平均
        low_m  = mask_region_0360(da_119, lat_119, lon_119, lon0, lon1, lat0, lat1)
        high_m = mask_region_0360(da_585, lat_585, lon_585, lon0, lon1, lat0, lat1)

        ts_low  = annual_mean(area_weighted_mean_masked(low_m,  lat_119))
        ts_high = annual_mean(area_weighted_mean_masked(high_m, lat_585))

        # 同一模型内两个情景尽量对齐年份（公平比较）
        ts_low, ts_high = xr.align(ts_low, ts_high, join="inner")

        years = ts_low["time"].dt.year.values
        yL = ts_low.values
        yH = ts_high.values

        # 画线：颜色按情景，线型按模型
        lab_low  = f"{mname} {SCENARIOS['ssp119']['label']}" if set_label else None
        lab_high = f"{mname} {SCENARIOS['ssp585']['label']}" if set_label else None

        ax.plot(years, yL, linewidth=1.8, linestyle=ls,
                color=SCENARIOS["ssp119"]["color"], label=lab_low)
        ax.plot(years, yH, linewidth=1.8, linestyle=ls,
                color=SCENARIOS["ssp585"]["color"], label=lab_high)

        # 斜率与R2（不画趋势线也能表达趋势；更清爽）
        mL, bL, r2L = linfit_r2(years, yL)
        mH, bH, r2H = linfit_r2(years, yH)

        txt_lines.append(
            f"{mname}: 119 slope={mL:.3g}/yr (R²={r2L:.2f}), "
            f"585 slope={mH:.3g}/yr (R²={r2H:.2f})"
        )

    ax.set_title(rname)
    ax.set_ylabel("zooc")

    ax.text(
        0.01, 0.02,
        "\n".join(txt_lines),
        transform=ax.transAxes, fontsize=9, va="bottom"
    )

# 图例放在第一幅或整张图顶部
axes[0].legend(loc="upper right", fontsize=9)
axes[-1].set_xlabel("Year")

plt.tight_layout()
out = OUTDIR / f"FigA_regional_ts_5regions_2models_2scenarios_{FIELD}_{YEAR0}-{YEAR1}.png"
plt.savefig(out, dpi=300)
plt.show()
print("✅ Saved:", out)
