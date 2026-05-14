import os
import time
import warnings
import joblib
import optuna
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

climate_path="input_data_path"
sif_path="input_data_path"
IGBP_path="input_data_path"
Soil_texture_path="input_data_path"
vege_mask_path="input_data_path"
slope_path="input_data_path"

out_dir='save_model_path'
os.makedirs(out_dir,exist_ok=True)

MODEL_FILENAME=os.path.join(out_dir,"01_XGB_model_grassland.joblib")
PLOT_FILENAME=os.path.join(out_dir,"01_XGB_model_grassland.png")
CSV_FILENAME=os.path.join(out_dir,"01_XGB_model_grassland.csv")

DYNAMIC_FEATURE_NAMES=["Air_temperature","Solar_radiation","VPD","Rain_frequency","Soil moisture"]
STATIC_FEATURE_NAMES=["Soil_clay","Soil_sand","SLOPE","IGBP"]
ALL_FEATURE_NAMES=DYNAMIC_FEATURE_NAMES+STATIC_FEATURE_NAMES

def load_data():
    climate=xr.open_dataset(climate_path+"ERA5_GS_Annual_Mean_2001-2021.nc").sel(year=slice("2001","2021"))
    vpd_ds=xr.open_dataset(climate_path+"VPD_GS_Annual_Mean_2001-2021.nc").sel(time=slice("2001-01-01","2021-12-31"))
    rain_freq_ds=xr.open_dataset(climate_path+"GPM_Frequency_GS_Annual_Mean_2001-2021.nc").sel(time=slice("2001-01","2021-12"))
    air_temperature=climate["t2m"]
    solar_radiation=climate["ssrd"]
    soil_moisture=climate["sm"].values
    VPD=vpd_ds["vpd"].values
    Rain_frequency=rain_freq_ds["precip_frequency"].values
    IGBP=xr.open_dataset(IGBP_path)["IGBP"].values
    Soil_clay=xr.open_dataset(Soil_texture_path+"T_CLAY_0.25deg.nc")["T_CLAY"].values
    Soil_sand=xr.open_dataset(Soil_texture_path+"T_SAND_0.25deg.nc")["T_SAND"].values
    Slope=xr.open_dataset(slope_path)["SLOPE"].values
    Sif=xr.open_dataset(sif_path)["clear_daily_SIF"].values
    vege_mask=xr.open_dataset(vege_mask_path)["sif_mask"].values
    return air_temperature,solar_radiation,VPD,Rain_frequency,soil_moisture,Soil_clay,Soil_sand,Slope,IGBP,Sif,vege_mask

def preprocess_data(air_temperature,solar_radiation,VPD,Rain_frequency,soil_moisture,Soil_clay,Soil_sand,Slope,IGBP,Sif,vege_mask):
    igbp_mask=np.isin(IGBP,[1,2,3,4,5,6,7,8,9,10])
    dynamic_features=np.stack([air_temperature,solar_radiation,VPD,Rain_frequency,soil_moisture],axis=-1)
    static_features=np.stack([Soil_clay,Soil_sand,Slope,IGBP],axis=-1)
    valid_mask=(~np.isnan(static_features).any(axis=-1))&(~np.isnan(dynamic_features).any(axis=(0,-1)))&(~np.isnan(Sif).any(axis=0))&((Sif>0).all(axis=0))&(igbp_mask)&(vege_mask==1)
    rows,cols=np.where(valid_mask)
    n_time,n_pixel=Sif.shape[0],len(rows)
    X_dynamic=dynamic_features[:,rows,cols,:].reshape(n_time*n_pixel,len(DYNAMIC_FEATURE_NAMES))
    X_static=np.tile(static_features[rows,cols,:],(n_time,1))
    y=Sif[:,rows,cols].reshape(-1)
    X=pd.DataFrame(np.concatenate([X_dynamic,X_static],axis=1),columns=ALL_FEATURE_NAMES)
    X["IGBP"]=X["IGBP"].round().astype(int).astype("category")
    meta=pd.DataFrame({
        "row":np.tile(rows,n_time),
        "col":np.tile(cols,n_time)})
    print(f"valid sample size: {len(X)}")
    return X,y,meta

def make_spatial_blocks(meta,block_size=20):
    return np.array([(str(r)+"_"+str(c)) for r,c in zip(meta["row"].values//block_size,meta["col"].values//block_size)])

def build_model(params):
    return xgb.XGBRegressor(objective="reg:squarederror",tree_method="hist",enable_categorical=True,eval_metric="rmse",n_jobs=-1,random_state=42,**params)

def objective(trial,X_train,y_train,groups_train):
    params={
        "n_estimators":trial.suggest_int("n_estimators",300,2500),
        "learning_rate":trial.suggest_float("learning_rate",0.005,0.15,log=True),
        "max_depth":trial.suggest_int("max_depth",3,12),
        "min_child_weight":trial.suggest_float("min_child_weight",1,30),
        "subsample":trial.suggest_float("subsample",0.6,1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree",0.6,1.0),
        "gamma":trial.suggest_float("gamma",1e-8,5.0,log=True),
        "reg_alpha":trial.suggest_float("reg_alpha",1e-8,10.0,log=True),
        "reg_lambda":trial.suggest_float("reg_lambda",1e-8,20.0,log=True)
    }

    gkf=GroupKFold(n_splits=5)
    rmse_list=[]

    for train_idx,val_idx in gkf.split(X_train,y_train,groups_train):
        X_tr,X_val=X_train.iloc[train_idx].copy(),X_train.iloc[val_idx].copy()
        y_tr,y_val=y_train[train_idx],y_train[val_idx]

        X_tr["IGBP"]=X_tr["IGBP"].astype("category")
        X_val["IGBP"]=X_val["IGBP"].astype("category")

        model=build_model(params)

        model.fit(
            X_tr,y_tr,
            eval_set=[(X_val,y_val)],
            early_stopping_rounds=50,
            verbose=False)

        pred=model.predict(X_val)
        rmse_list.append(np.sqrt(mean_squared_error(y_val,pred)))

    return np.mean(rmse_list)

def train_xgb_model(X,y,meta,n_trials=100,block_size=20):
    print("model training...")
    groups=make_spatial_blocks(meta,block_size)

    outer_split=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    train_idx,test_idx=next(outer_split.split(X,y,groups))

    X_train_full,X_test=X.iloc[train_idx].copy(),X.iloc[test_idx].copy()
    y_train_full,y_test=y[train_idx],y[test_idx]
    groups_train=groups[train_idx]

    X_train_full["IGBP"]=X_train_full["IGBP"].astype("category")
    X_test["IGBP"]=X_test["IGBP"].astype("category")

    study=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial,X_train_full,y_train_full,groups_train),n_trials=n_trials)

    print(f" RMSE: {study.best_value:.4f}")

    best_params=study.best_params.copy()

    inner_split=GroupShuffleSplit(n_splits=1,test_size=0.15,random_state=24)
    subtrain_idx,val_idx=next(inner_split.split(X_train_full,y_train_full,groups_train))

    X_subtrain,X_val=X_train_full.iloc[subtrain_idx].copy(),X_train_full.iloc[val_idx].copy()
    y_subtrain,y_val=y_train_full[subtrain_idx],y_train_full[val_idx]

    X_subtrain["IGBP"]=X_subtrain["IGBP"].astype("category")
    X_val["IGBP"]=X_val["IGBP"].astype("category")

    temp_model=build_model(best_params)

    temp_model.fit(
        X_subtrain,y_subtrain,
        eval_set=[(X_val,y_val)],
        early_stopping_rounds=50,
        verbose=False)

    best_iteration=temp_model.best_iteration+1 if temp_model.best_iteration is not None else best_params["n_estimators"]

    final_params=best_params.copy()
    final_params["n_estimators"]=best_iteration

    final_model=build_model(final_params)
    final_model.fit(X_train_full,y_train_full,verbose=False)

    joblib.dump(final_model,MODEL_FILENAME)
    print(f"模型已保存: {MODEL_FILENAME}")

    y_pred=final_model.predict(X_test)

    return y_test,y_pred

def plot_performance(y_test,y_pred):
    r2=r2_score(y_test,y_pred)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    mae=mean_absolute_error(y_test,y_pred)

    fig,ax=plt.subplots(figsize=(8,8))
    ax.scatter(y_test,y_pred,s=12,alpha=0.25,edgecolors="none")

    lim_min=min(np.nanmin(y_test),np.nanmin(y_pred))
    lim_max=max(np.nanmax(y_test),np.nanmax(y_pred))

    ax.plot([lim_min,lim_max],[lim_min,lim_max],"k--",linewidth=1.5)

    ax.set_xlim(lim_min,lim_max)
    ax.set_ylim(lim_min,lim_max)

    ax.set_xlabel("Observed SIF",fontsize=14)
    ax.set_ylabel("Predicted SIF",fontsize=14)

    ax.text(
        0.05,0.95,
        f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}\nMAE={mae:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round",fc="white",alpha=0.8)
    )
    plt.tight_layout()
    plt.savefig(PLOT_FILENAME,dpi=300)
    pd.DataFrame({"y_test":y_test,"y_pred":y_pred}).to_csv(CSV_FILENAME,index=False)
    print(f"R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")

if __name__=="__main__":
    start_time=time.time()
    data=load_data()
    X,y,meta=preprocess_data(*data)
    y_test,y_pred=train_xgb_model(X,y,meta,n_trials=100,block_size=20)
    plot_performance(y_test,y_pred)
    end_time=time.time()
