import xarray as xr
from pathlib import Path
import numpy as np

DIR_HIST = Path(r"F:\1A_Files\Zooc\historic")
DIR_LOW  = Path(r"F:\1A_Files\Zooc\ssp119")
DIR_HIGH = Path(r"F:\1A_Files\Zooc\ssp585")
OUTDIR   = Path(r"F:\1A_Files\Zooc\combined")
OUTDIR.mkdir(parents=True, exist_ok=True)

VAR_NAME = "zooc"   # 如果不是 zooc，改这里

def combine_folder(folder: Path, out_nc: Path):
    files = sorted(folder.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No nc files in {folder}")

    # ✅ 合并多个文件：按 time 拼接；use_cftime 适配 360_day
    ds = xr.open_mfdataset(
        [str(f) for f in files],
        combine="by_coords",
        use_cftime=True,
        parallel=False
    )

    # 只保留目标变量（可选）
    if VAR_NAME in ds.data_vars:
        ds = ds[[VAR_NAME]]
    else:
        # 如果变量名不叫 zooc，就保留全部并提示
        print(f"Warning: {VAR_NAME} not found, kept all data_vars:", list(ds.data_vars)[:10])

    # ✅ 时间排序 + 去重（有些文件边界年份会重叠）
    ds = ds.sortby("time")
    t = ds["time"].values
    _, idx = np.unique(t, return_index=True)
    ds = ds.isel(time=idx)

    # 保存
    ds.to_netcdf(out_nc)
    print("Saved:", out_nc)
    print("Time span:", str(ds.time.values[0]), "->", str(ds.time.values[-1]))
    return out_nc

combine_folder(DIR_HIST, OUTDIR / "zooc_historical_combined.nc")
combine_folder(DIR_LOW,  OUTDIR / "zooc_ssp119_combined.nc")
combine_folder(DIR_HIGH, OUTDIR / "zooc_ssp585_combined.nc")
