import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

print("\n=== 5 Regions × 2 Models × 2 Scenarios (% change vs historical) ===\n")

# ===================== 路径 =====================
MODELS = {
    "UKESM": {
        "historic": Path(r"F:\1A_Files\Zooc\historic"),
        "ssp119":   Path(r"F:\1A_Files\Zooc\ssp119"),
        "ssp585":   Path(r"F:\1A_Files\Zooc\ssp585"),
    },
    "IPSL": {
        "historic": Path(r"F:\1A_Files\Zooc_IPSL-CM6A-LR\historic"),
        "ssp119":   Path(r"F:\1A_Files\Zooc_IPSL-CM6A-LR\ssp119"),
        "ssp585":   Path(r"F:\1A_Files\Zooc_IPSL-CM6A-LR\ssp585"),
    }
}

OUTDIR = Path(r"F:\1A_Files\Zooc\figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

FIELD = "surface"
VAR_NAME = "zooc"

BASE_START, BASE_END = 1995, 2014
END_START, END_END   = 2081, 2100

# ===================== 5 区域 =====================
REGIONS = [
    ("North Atlantic (high lat)", -60, -10, 45, 65),
    ("Arabian Sea",                50,  75,  5, 25),
    ("South China Sea",           105, 120,  0, 25),
    ("Southern Ocean",              0, 360, -60, -40),
    ("Equatorial Pacific",       -170, -120, -5, 5),
]

# ===================== 工具函数 =====================

def open_da_mf(folder):
    files = sorted(folder.glob("*.nc"))
    ds = xr.open_mfdataset(
        [str(f) for f in files],
        combine="by_coords",
        use_cftime=True,
        chunks={"time": 12}
    )
    var = VAR_NAME if VAR_NAME in ds.data_vars else list(ds.data_vars.keys())[0]
    return ds[var].sortby("time")

def select_surface(da):
    for d in ["lev","depth","olevel","deptht","z_t"]:
        if d in da.dims:
            return da.isel({d:0})
    return da

def find_latlon(da):
    for latname in ["lat","latitude","nav_lat","TLAT","yt_ocean"]:
        if latname in da.coords:
            lat = da[latname]; break
    for lonname in ["lon","longitude","nav_lon","TLONG","xt_ocean"]:
        if lonname in da.coords:
            lon = da[lonname]; break
    return lat, lon

def mask_region(da, lat, lon, lon0, lon1, lat0, lat1):
    if lon.max() > 180:
        lon0 = lon0 % 360
        lon1 = lon1 % 360
    cond = (lon >= min(lon0,lon1)) & (lon <= max(lon0,lon1)) \
           & (lat >= lat0) & (lat <= lat1)
    return da.where(cond)

def area_mean(da, lat):
    w = np.cos(np.deg2rad(lat))
    dims = [d for d in da.dims if d!="time"]
    return da.weighted(w).mean(dims)

def window_mean(ts, y0, y1):
    yrs = ts["time"].dt.year
    return ts.where((yrs>=y0)&(yrs<=y1), drop=True).mean()

# ===================== 读取全部模型数据 =====================

DATA = {}

for m in MODELS:
    DATA[m] = {}
    for exp in MODELS[m]:
        da = select_surface(open_da_mf(MODELS[m][exp]))
        lat, lon = find_latlon(da)
        DATA[m][exp] = {"da":da, "lat":lat, "lon":lon}

# ===================== 计算百分比 =====================

results = {}

for m in DATA:
    results[m] = {"ssp119":[], "ssp585":[]}
    for (name, lon0, lon1, lat0, lat1) in REGIONS:

        # baseline
        da_h = DATA[m]["historic"]["da"]
        lat = DATA[m]["historic"]["lat"]
        lon = DATA[m]["historic"]["lon"]

        base_mask = mask_region(da_h, lat, lon, lon0, lon1, lat0, lat1)
        base_ts = area_mean(base_mask, lat)
        base = window_mean(base_ts, BASE_START, BASE_END)

        for scen in ["ssp119","ssp585"]:
            da_s = DATA[m][scen]["da"]
            lat_s = DATA[m][scen]["lat"]
            lon_s = DATA[m][scen]["lon"]

            mask = mask_region(da_s, lat_s, lon_s, lon0, lon1, lat0, lat1)
            ts = area_mean(mask, lat_s)
            end = window_mean(ts, END_START, END_END)

            pct = (end - base) / base * 100
            results[m][scen].append(float(pct.compute().values))

# ===================== 画图 =====================

regions_names = [r[0] for r in REGIONS]
x = np.arange(len(regions_names))
width = 0.18

fig, ax = plt.subplots(figsize=(14,6))

offset = {
    ("UKESM","ssp119"): -1.5*width,
    ("UKESM","ssp585"): -0.5*width,
    ("IPSL","ssp119"):  0.5*width,
    ("IPSL","ssp585"):  1.5*width,
}

colors = {"ssp119":"tab:blue", "ssp585":"tab:orange"}
alpha  = {"UKESM":1.0, "IPSL":0.6}

for m in results:
    for scen in results[m]:
        vals = results[m][scen]
        ax.bar(
            x + offset[(m,scen)],
            vals,
            width,
            label=f"{m}-{scen}",
            color=colors[scen],
            alpha=alpha[m]
        )

ax.axhline(0,color="black",linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(regions_names, rotation=20)
ax.set_ylabel("% change (2081–2100 vs 1995–2014)")
ax.set_title("Regional Zooplankton Carbon Change\n5 Regions × 2 Models × 2 Scenarios")

ax.legend(fontsize=9)
plt.tight_layout()

out = OUTDIR / "5regions_2models_percent_change.png"
plt.savefig(out, dpi=300)
plt.show()

print("✅ Saved:", out)
