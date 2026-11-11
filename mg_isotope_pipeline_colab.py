
!pip -q install xgboost lightgbm catboost optuna shap scikit-learn openpyxl 2>/dev/null

import numpy as np, pandas as pd, re, warnings, json
warnings.filterwarnings("ignore")
from google.colab import files
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna, shap
import matplotlib.pyplot as plt
optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGET, IDCOL, CATCOL = "d26Mg", "Sample_ID", "Rock_Type"

# ---------------------------------------------------------------- 1. Loading the files  -----
up = files.upload()
fn = list(up.keys())[0]
df = pd.read_excel(fn)
n0 = len(df)

# --------------------------------------------------------- 2. CLEAN the dataset -----------
# 'X' (and friends) are missing-value tokens in this dataset -> NaN, then numeric
df = df.replace(["X","x","?","NA","N/A","na","n.d.","nd","-","","bdl","BDL","-999",-999], np.nan)
for c in df.columns:
    if c not in [IDCOL, CATCOL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

base_num = [c for c in df.columns if c not in [TARGET,IDCOL,CATCOL,"Total"]
            and pd.api.types.is_numeric_dtype(df[c])]
# drop samples missing >60% of features, and columns missing >50%
df = df[df[base_num].isna().mean(axis=1) <= 0.60].reset_index(drop=True)
miss = df[base_num].isna().mean()
base_num = [c for c in base_num if miss[c] <= 0.50]
y = df[TARGET].values
print(f"Rows {n0} -> {len(df)} | features kept: {len(base_num)} | "
      f"d26Mg sigma = {y.std():.3f} (baseline to beat)")

# ------------------------------------- 3. FEATURE ENGINEERING (computable) ----
# Only theory-motivated features that CAN be derived from bulk major/trace data.
# (1/T^2, Rayleigh-f, and growth-rate Peclet are NOT derivable here and omitted.)
MW = dict(SiO2=60.08,TiO2=79.87,Al2O3=101.96,Fe2O3T=159.69,MnO=70.94,
          MgO=40.30,CaO=56.08,Na2O=61.98,K2O=94.20,P2O5=141.94)
def molar(ox): return df[ox]/MW[ox] if ox in df else None
CH = dict(La=0.237,Ce=0.612,Pr=0.095,Nd=0.467,Sm=0.153,Eu=0.058,
          Gd=0.2055,Tb=0.0374,Dy=0.254,Yb=0.170,Lu=0.0254)   # chondrite (Sun & McDonough '89)
def cn(el): return df[el]/CH[el] if el in df else None
S = set(df.columns); eng = {}
if {"CaO","MgO"}<=S:            eng["Ca_Mg"]=df["CaO"]/df["MgO"].replace(0,np.nan)      # carbonate control
if {"MgO","Fe2O3T"}<=S:        eng["Mg_number"]=molar("MgO")/(molar("MgO")+molar("Fe2O3T")*2)  # bond environ.
if {"Al2O3","SiO2"}<=S:        eng["Al_Si"]=df["Al2O3"]/df["SiO2"].replace(0,np.nan)   # coordination
if {"MgO","SiO2"}<=S:          eng["Mg_Si"]=df["MgO"]/df["SiO2"].replace(0,np.nan)
if {"Al2O3","CaO","Na2O","K2O"}<=S:                                                    # CIA (weathering)
    Al,Ca,Na,K = molar("Al2O3"),molar("CaO"),molar("Na2O"),molar("K2O")
    eng["CIA"]=100*Al/(Al+Ca+Na+K)
if {"La","Yb"}<=S:  eng["La_Yb"]=df["La"]/df["Yb"].replace(0,np.nan)                   # partial-melting proxy
if {"Sm","Nd"}<=S:  eng["Sm_Nd"]=df["Sm"]/df["Nd"].replace(0,np.nan)
if {"La","Sm"}<=S:  eng["La_Sm"]=df["La"]/df["Sm"].replace(0,np.nan)
if {"Rb","Sr"}<=S:  eng["Rb_Sr"]=df["Rb"]/df["Sr"].replace(0,np.nan)                   # fluid-rock
if {"Sr","Y"}<=S:   eng["Sr_Y"]=df["Sr"]/df["Y"].replace(0,np.nan)
if {"Ba","Rb"}<=S:  eng["Ba_Rb"]=df["Ba"]/df["Rb"].replace(0,np.nan)
if {"V","Sc"}<=S:   eng["V_Sc"]=df["V"]/df["Sc"].replace(0,np.nan)                     # redox (indirect)
if {"Th","U"}<=S:   eng["Th_U"]=df["Th"]/df["U"].replace(0,np.nan)
if {"Eu","Sm","Gd"}<=S: eng["Eu_anom"]=cn("Eu")/np.sqrt(cn("Sm")*cn("Gd"))            # Eu/Eu*
if {"Ce","La","Pr"}<=S: eng["Ce_anom"]=cn("Ce")/np.sqrt(cn("La")*cn("Pr"))           # Ce/Ce*
if {"Na2O","K2O"}<=S:   eng["alkali"]=df["Na2O"]+df["K2O"]
for k,v in eng.items():
    if v is not None: df[k]=v.replace([np.inf,-np.inf],np.nan)
eng_cols=[k for k in eng]; print(f"Engineered {len(eng_cols)} features:", eng_cols)

# --------------------------------------------- 4. REGIME (feature + grouping) --
rt = df[CATCOL].astype(str).str.lower()
def regime(s):
    if re.search(r"carbonatit|siderite|dolomit|calcite|limestone|marble", s): return "carbonatite"
    if re.search(r"weather|saprolit|soil|laterite|regolith|sediment|shale|loess|clay|chert|bauxite", s): return "sedimentary"
    if re.search(r"eclogit|amphibolit|granulit|schist|gneiss|greenschist|migmatit|metaperid|roding", s): return "metamorphic"
    if re.search(r"morb|basalt|oib|lamproit|granit|diabas|xenolith|lherzolit|peridotit|gabbro|rhyolit|"
                 r"andesit|trachy|ultrapotassic|kimberlit|dunit|pyroxenit|syenit|dolerit|picrit|komatiit|tholeiit", s): return "igneous"
    return "other"
df["Regime"]=rt.map(regime)

# --------------------------------------------------------- 5. BUILD X ---------
feat_num = base_num + eng_cols
X = pd.concat([df[feat_num], pd.get_dummies(df["Regime"], prefix="reg")], axis=1)
feat_cols = X.columns.tolist()
for c in feat_num:                                            # log raw concentrations only
    if not re.search(r"_|#|CIA|anom|alkali|number|reg_", c) and (X[c].dropna()>=0).all():
        X[c]=np.log1p(X[c])
X_imp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=feat_cols)
groups = df[CATCOL].astype(str).values
kf, gkf = KFold(5,shuffle=True,random_state=42), GroupKFold(5)
def sc(a,b): return dict(R2=r2_score(a,b),RMSE=np.sqrt(mean_squared_error(a,b)),MAE=mean_absolute_error(a,b))

# ------------------------------------------------- 6.COMPARISON of diffrent models --------
models = [
 ("Ridge",        Pipeline([("s",StandardScaler()),("m",Ridge(alpha=10))]), X_imp),
 ("RandomForest", RandomForestRegressor(300,random_state=42,n_jobs=-1), X_imp),
 ("XGBoost",      XGBRegressor(n_estimators=400,learning_rate=0.05,max_depth=4,
                    subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1), X),
 ("LightGBM",     LGBMRegressor(n_estimators=500,learning_rate=0.05,random_state=42,verbose=-1,n_jobs=-1), X),
 ("CatBoost",     CatBoostRegressor(iterations=400,learning_rate=0.05,depth=5,verbose=0,
                    random_state=42,thread_count=-1), X),
 ("NeuralNet",    Pipeline([("s",StandardScaler()),
                    ("m",MLPRegressor((128,64),max_iter=400,early_stopping=True,random_state=42))]), X_imp),
]
rows=[]
for name,mdl,Xin in models:
    pr=cross_val_predict(mdl,Xin,y,cv=kf)
    pg=cross_val_predict(mdl,Xin,y,cv=gkf,groups=groups)
    r,g=sc(y,pr),sc(y,pg)
    rows.append(dict(model=name,R2_rand=r["R2"],RMSE_rand=r["RMSE"],
                     R2_group=g["R2"],RMSE_group=g["RMSE"],MAE_group=g["MAE"]))
    print(f"  {name} done")
board=pd.DataFrame(rows).sort_values("R2_group",ascending=False).reset_index(drop=True)
pd.set_option("display.float_format",lambda v:f"{v:.3f}")
print("\n=== MODEL COMPARISON (rand = optimistic ; group = honest) ===")
print(board.to_string(index=False))
best=board.iloc[0]["model"]

# ------------------------------------------------- 7. OPTUNA  -----
print(f"\nOptuna tuning {best} ...")
g3=GroupKFold(3)
def cvr(m,Xin,cv): return np.sqrt(mean_squared_error(y,cross_val_predict(m,Xin,y,cv=cv,groups=groups)))
def objective(t):
    if best=="CatBoost":
        m=CatBoostRegressor(verbose=0,random_state=42,thread_count=-1,
            iterations=t.suggest_int("iterations",200,600,step=50),
            depth=t.suggest_int("depth",4,8),
            learning_rate=t.suggest_float("learning_rate",0.02,0.15,log=True),
            l2_leaf_reg=t.suggest_float("l2_leaf_reg",1,6),
            subsample=t.suggest_float("subsample",0.6,1.0)); Xin=X
    elif best=="XGBoost":
        m=XGBRegressor(random_state=42,n_jobs=-1,
            n_estimators=t.suggest_int("n_estimators",300,900,step=100),
            max_depth=t.suggest_int("max_depth",3,7),
            learning_rate=t.suggest_float("learning_rate",0.01,0.15,log=True),
            subsample=t.suggest_float("subsample",0.6,1.0),
            colsample_bytree=t.suggest_float("colsample_bytree",0.6,1.0),
            reg_lambda=t.suggest_float("reg_lambda",0.5,5)); Xin=X
    elif best=="LightGBM":
        m=LGBMRegressor(random_state=42,verbose=-1,n_jobs=-1,
            n_estimators=t.suggest_int("n_estimators",300,1000,step=100),
            num_leaves=t.suggest_int("num_leaves",15,80),
            learning_rate=t.suggest_float("learning_rate",0.01,0.15,log=True),
            subsample=t.suggest_float("subsample",0.6,1.0)); Xin=X
    else:
        m=RandomForestRegressor(random_state=42,n_jobs=-1,
            n_estimators=t.suggest_int("n_estimators",300,900,step=100),
            max_depth=t.suggest_categorical("max_depth",[None,8,12,18]),
            min_samples_leaf=t.suggest_int("min_samples_leaf",1,5)); Xin=X_imp
    return cvr(m,Xin,g3)
study=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective,n_trials=40,show_progress_bar=True)
print("Best params:",study.best_params)

bp=study.best_params
if best=="CatBoost": bm=CatBoostRegressor(verbose=0,random_state=42,thread_count=-1,**bp); Xin=X
elif best=="XGBoost": bm=XGBRegressor(random_state=42,n_jobs=-1,**bp); Xin=X
elif best=="LightGBM": bm=LGBMRegressor(random_state=42,verbose=-1,n_jobs=-1,**bp); Xin=X
else: bm=RandomForestRegressor(random_state=42,n_jobs=-1,**bp); Xin=X_imp

final=cross_val_predict(bm,Xin,y,cv=gkf,groups=groups)
fs=sc(y,final)
print(f"\n=== TUNED {best} — HONEST (held-out rock types) ===")
print(f"R2={fs['R2']:.3f}  RMSE={fs['RMSE']:.3f}  MAE={fs['MAE']:.3f}  (sigma={y.std():.3f})")

# ------------------------------------------------- 8. PER-REGIME --------------
df["pred"]=final; df["err"]=final-y
print("\n=== PER-REGIME (where does it actually work?) ===")
for reg,gg in df.groupby("Regime"):
    if len(gg)>=15:
        s=sc(gg[TARGET].values,gg["pred"].values)
        print(f"  {reg:12s} n={len(gg):4d}  R2={s['R2']:.3f}  RMSE={s['RMSE']:.3f}  MAE={s['MAE']:.3f}")
worst=df.reindex(df["err"].abs().sort_values(ascending=False).index).head(25)
print("\nWorst-error rock types (process-dominated anomalies):")
print(worst["Rock_Type"].value_counts().head(10).to_string())

# ------------------------------------------------- 9. SHAP (Knowing the cause) + PLOTS ------------
bm.fit(Xin,y)
expl=shap.TreeExplainer(bm); sv=expl.shap_values(Xin)
shap.summary_plot(sv,Xin,plot_type="bar",max_display=15,show=False)
plt.tight_layout(); plt.savefig("shap_importance.png",dpi=150,bbox_inches="tight"); plt.show()

plt.figure(figsize=(5,5))
for reg,gg in df.groupby("Regime"):
    plt.scatter(gg[TARGET],gg["pred"],s=12,alpha=.5,label=reg)
lim=[y.min()-.2,y.max()+.2]; plt.plot(lim,lim,"r--")
plt.xlabel("Measured δ²⁶Mg (‰)"); plt.ylabel("Predicted δ²⁶Mg (‰)")
plt.legend(fontsize=7); plt.title(f"{best} tuned — honest R²={fs['R2']:.2f}")
plt.tight_layout(); plt.savefig("pred_vs_meas.png",dpi=150); plt.show()

# ------------------------------------------------- 10. Saving all the results -------------------
board.to_csv("model_comparison.csv",index=False)
df[[IDCOL,CATCOL,"Regime",TARGET,"pred","err"]].to_csv("predictions.csv",index=False)
json.dump({"n_samples":int(len(y)),"n_features":int(len(feat_cols)),
           "best_model":best,"tuned":{k:round(float(v),3) for k,v in fs.items()},
           "sigma":round(float(y.std()),3),"best_params":bp},
          open("run_summary.json","w"),indent=2)
for f in ["model_comparison.csv","predictions.csv","run_summary.json",
          "shap_importance.png","pred_vs_meas.png"]:
    files.download(f)
print("\nSaved: model_comparison.csv, predictions.csv, run_summary.json, shap + scatter PNGs")
