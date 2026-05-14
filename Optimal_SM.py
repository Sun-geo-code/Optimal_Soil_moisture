import os, time, joblib, warnings
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from tqdm import tqdm
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

climate_path="input_data_path"
sif_path="input_data_path"
IGBP_path="input_data_path"
Soil_texture_path="input_data_path"
vege_mask_path="input_data_path"
slope_path="input_data_path"

MODEL_FILENAME = r"save_model_path"
OUTPUT_NC_PATH = r"output_path"

SM_CANDIDATES = np.arange(0.01, 0.8 + 1e-5, 0.01)
FEATURE_NAMES = ["Air_temperature", "Solar_radiation", "VPD", "Rain_frequency", "Soil moisture", "Soil_clay", "Soil_sand", "SLOPE", "IGBP"]

def standardize_dims(ds):
    rename_dict = {}
    for dim in ds.dims:
        d = str(dim).lower()
        if d in ["latitude", "lat"] and dim != "lat": rename_dict[dim] = "lat"
        elif d in ["longitude", "lon"] and dim != "lon": rename_dict[dim] = "lon"
        elif d in ["year", "time", "t"] and dim != "time": rename_dict[dim] = "time"
    return ds.rename(rename_dict) if rename_dict else ds


def classify_sm_sif_response(sm_candidates, sif_curve, flat_fraction=0.03, threshold_change=0.03):
    n = len(sm_candidates)
    if n < 5: return 1

    win_len = max(5, n // 5)
    if win_len % 2 == 0: win_len += 1
    if win_len >= n: win_len = n - 1 if n % 2 == 0 else n

    try:
        sif_smooth = savgol_filter(sif_curve, window_length=win_len, polyorder=2)
    except Exception:
        sif_smooth = sif_curve

    sif_max, sif_min = np.max(sif_smooth), np.min(sif_smooth)
    sif_range, sif_mean = sif_max - sif_min, np.mean(sif_smooth)
    threshold_abs = np.abs(sif_mean) * flat_fraction if np.abs(sif_mean) > 1e-6 else 1e-5

    if sif_range < threshold_abs: return 1

    idx_max, idx_min = np.argmax(sif_smooth), np.argmin(sif_smooth)
    val_max, val_min = sif_smooth[idx_max], sif_smooth[idx_min]
    val_start, val_end = sif_smooth[0], sif_smooth[-1]
    norm_scale = sif_range if sif_range > 1e-6 else 1.0

    rise_from_start = (val_max - val_start) / norm_scale
    drop_to_end = (val_max - val_end) / norm_scale
    total_trend = (val_end - val_start) / norm_scale
    drop_from_start = (val_start - val_min) / norm_scale
    rise_to_end = (val_end - val_min) / norm_scale

    def is_tail_flat(data, total_range, fraction=0.3, tolerance=0.10):
        n_tail = int(len(data) * fraction)
        if n_tail < 2: return False
        return (np.max(data[-n_tail:]) - np.min(data[-n_tail:])) < total_range * tolerance

    if 0 < idx_max < n - 1 and rise_from_start > threshold_change and drop_to_end > 0.05: return 2
    if total_trend > threshold_change: return 4 if is_tail_flat(sif_smooth, sif_range) else 3
    if (drop_from_start > threshold_change and rise_to_end > 0.05) or total_trend < -threshold_change: return 5
    return 1


def load_static_data():
    IGBP = standardize_dims('input_data')["IGBP"].values
    Soil_clay = standardize_dims('input_data')["T_CLAY"].values
    Soil_sand = standardize_dims('input_data')["T_SAND"].values
    Slope = standardize_dims('input_data')["SLOPE"].values
    vege_mask = standardize_dims('input_data')["sif_mask"].values
    return IGBP, Soil_clay, Soil_sand, Slope, vege_mask
def open_dynamic_data():
    ds_clim = standardize_dims('climate_data')
    ds_vpd = standardize_dims('climate_data')
    ds_rain_freq = standardize_dims('climate_data')
    ds_sif = standardize_dims('climate_data')
    return ds_clim, ds_vpd, ds_rain_freq, ds_sif


def build_feature_dataframe(t2m, ssrd, vpd, freq, sm, clay, sand, slope, igbp):
    df = pd.DataFrame({
        "Air_temperature": t2m.ravel(),
        "Solar_radiation": ssrd.ravel(),
        "VPD": vpd.ravel(),
        "Rain_frequency": freq.ravel(),
        "Soil moisture": sm.ravel(),
        "Soil_clay": clay.ravel(),
        "Soil_sand": sand.ravel(),
        "SLOPE": slope.ravel(),
        "IGBP": igbp.ravel()
    })
    df["IGBP"] = df["IGBP"].astype(int).astype("category")
    return df[FEATURE_NAMES]


def process_lat_row(lat_idx, model, static_data, years, ds_clim, ds_vpd, ds_rain_freq, ds_sif):
    IGBP_all, Clay_all, Sand_all, Slope_all, Mask_all = static_data
    igbp_row, clay_row, sand_row, slope_row, mask_row = IGBP_all[lat_idx], Clay_all[lat_idx], Sand_all[lat_idx], Slope_all[lat_idx], Mask_all[lat_idx]
    valid_lons = np.where((mask_row == 1) & (igbp_row > 0))[0]

    if len(valid_lons) == 0: return None

    try:
        t2m = ds_clim["t2m"].isel(lat=lat_idx).values
        ssrd = ds_clim["ssrd"].isel(lat=lat_idx).values / (24 * 3600)
        vpd = ds_vpd["vpd"].isel(lat=lat_idx).values
        freq = ds_rain_freq["precip_frequency"].isel(lat=lat_idx).values
        swvl1 = ds_clim["swvl1"].isel(lat=lat_idx).values
        swvl2 = ds_clim["swvl2"].isel(lat=lat_idx).values
        sif_obs = ds_sif["clear_daily_SIF"].isel(lat=lat_idx).values
        sm = (7 * swvl1 + 21 * swvl2) / 28
    except Exception as e:
        print(f"Error reading latitude index {lat_idx}: {e}")
        return None

    n_lons, n_years = len(igbp_row), len(years)
    res_pattern = np.full(n_lons, -1, dtype=np.int8)
    res_opt_sm = np.full((n_years, n_lons), np.nan, dtype=np.float32)
    res_max_sif = np.full((n_years, n_lons), np.nan, dtype=np.float32)
    res_orig_obs = np.full((n_years, n_lons), np.nan, dtype=np.float32)
    res_orig_pred = np.full((n_years, n_lons), np.nan, dtype=np.float32)
    res_orig_sm = np.full((n_years, n_lons), np.nan, dtype=np.float32)
    res_gain = np.full((n_years, n_lons), np.nan, dtype=np.float32)

    for i in range(0, len(valid_lons), 50):
        batch_lons = valid_lons[i:i + 50]
        n_batch = len(batch_lons)

        b_t2m, b_ssrd, b_vpd = t2m[:, batch_lons], ssrd[:, batch_lons], vpd[:, batch_lons]
        b_freq, b_sm, b_sif = freq[:, batch_lons], sm[:, batch_lons], sif_obs[:, batch_lons]
        b_clay = np.tile(clay_row[batch_lons], (n_years, 1))
        b_sand = np.tile(sand_row[batch_lons], (n_years, 1))
        b_slope = np.tile(slope_row[batch_lons], (n_years, 1))
        b_igbp = np.tile(igbp_row[batch_lons], (n_years, 1))

        df_base = build_feature_dataframe(b_t2m, b_ssrd, b_vpd, b_freq, b_sm, b_clay, b_sand, b_slope, b_igbp)
        pred_orig = model.predict(df_base).reshape(n_years, n_batch)

        res_orig_obs[:, batch_lons] = b_sif
        res_orig_pred[:, batch_lons] = pred_orig
        res_orig_sm[:, batch_lons] = b_sm

        predictions = np.empty((n_years, n_batch, len(SM_CANDIDATES)), dtype=np.float32)
        pred_df = df_base.copy()

        for sm_idx, sm_val in enumerate(SM_CANDIDATES):
            pred_df["Soil moisture"] = sm_val
            predictions[:, :, sm_idx] = model.predict(pred_df).reshape(n_years, n_batch)

        median_curves = np.nanmedian(predictions, axis=0)

        for k, lon_idx in enumerate(batch_lons):
            curve = median_curves[k]
            if not np.all(np.isnan(curve)): res_pattern[lon_idx] = classify_sm_sif_response(SM_CANDIDATES, curve)

        max_sif = np.max(predictions, axis=2)
        opt_sm = SM_CANDIDATES[np.argmax(predictions, axis=2)]
        gain = max_sif - pred_orig

        res_max_sif[:, batch_lons] = max_sif
        res_opt_sm[:, batch_lons] = opt_sm
        res_gain[:, batch_lons] = gain

    return res_pattern, res_opt_sm, res_max_sif, res_orig_obs, res_orig_pred, res_orig_sm, res_gain


def save_to_netcdf(output_path, years, lats, lons, out_pattern, out_opt_sm, out_max_sif, out_orig_obs, out_orig_pred, out_orig_sm, out_gain):
    ds_out = xr.Dataset(
        {
            "response_pattern": (("lat", "lon"), out_pattern),
            "optimal_sm": (("time", "lat", "lon"), out_opt_sm),
            "max_potential_sif": (("time", "lat", "lon"), out_max_sif),
            "original_sif_observed": (("time", "lat", "lon"), out_orig_obs),
            "original_sif_predicted": (("time", "lat", "lon"), out_orig_pred),
            "original_sm": (("time", "lat", "lon"), out_orig_sm),
            "sif_gain_model": (("time", "lat", "lon"), out_gain)
        },
        coords={"time": pd.to_datetime([f"{y}-01-01" for y in years]), "lat": lats, "lon": lons}
    )

    ds_out["response_pattern"].attrs = {
        "long_name": "SM-SIF response pattern based on multi-year median prediction curves",
        "description": "1=Flat, 2=Peak, 3=Increasing, 4=Increasing with saturation, 5=Negative",
        "flag_values": [1, 2, 3, 4, 5]
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds_out.to_netcdf(output_path)


def main():
    start_time = time.time()

    print(f"Loading XGBoost model: {MODEL_FILENAME}")
    model = xgb.XGBRegressor(enable_categorical=True)
    model = joblib.load(MODEL_FILENAME)

    print("Loading static data...")
    static_data = load_static_data()

    print("Opening dynamic datasets...")
    ds_clim, ds_vpd, ds_rain_freq, ds_sif = open_dynamic_data()

    years = list(range(2001, 2022))
    lats, lons = ds_sif.lat.values, ds_sif.lon.values
    n_years, n_lats, n_lons = len(years), len(lats), len(lons)

    out_pattern = np.full((n_lats, n_lons), -1, dtype=np.int8)
    out_opt_sm = np.full((n_years, n_lats, n_lons), np.nan, dtype=np.float32)
    out_max_sif = np.full((n_years, n_lats, n_lons), np.nan, dtype=np.float32)
    out_orig_obs = np.full((n_years, n_lats, n_lons), np.nan, dtype=np.float32)
    out_orig_pred = np.full((n_years, n_lats, n_lons), np.nan, dtype=np.float32)
    out_orig_sm = np.full((n_years, n_lats, n_lons), np.nan, dtype=np.float32)
    out_gain = np.full((n_years, n_lats, n_lons), np.nan, dtype=np.float32)

    print(f"Processing grid: time={n_years}, lat={n_lats}, lon={n_lons}")

    for lat_idx in tqdm(range(n_lats)):
        result = process_lat_row(lat_idx, model, static_data, years, ds_clim, ds_vpd, ds_rain_freq, ds_sif)
        if result is None: continue

        r_pat, r_opt_sm, r_max, r_obs, r_pred, r_sm, r_gain = result
        out_pattern[lat_idx] = r_pat
        out_opt_sm[:, lat_idx] = r_opt_sm
        out_max_sif[:, lat_idx] = r_max
        out_orig_obs[:, lat_idx] = r_obs
        out_orig_pred[:, lat_idx] = r_pred
        out_orig_sm[:, lat_idx] = r_sm
        out_gain[:, lat_idx] = r_gain

    print("Saving NetCDF output...")
    save_to_netcdf(OUTPUT_NC_PATH, years, lats, lons, out_pattern, out_opt_sm, out_max_sif, out_orig_obs, out_orig_pred, out_orig_sm, out_gain)

    for ds in [ds_clim, ds_vpd, ds_rain_freq, ds_sif]: ds.close()

    valid_pixels = np.sum(out_pattern > 0)
    print(f"Total pixels: {n_lats * n_lons}; valid pixels: {valid_pixels}")
    print(f"Completed in {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
