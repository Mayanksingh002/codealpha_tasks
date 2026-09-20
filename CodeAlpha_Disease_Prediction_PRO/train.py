import os,json,numpy as np,pandas as pd
from joblib import dump
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
SEED=42;os.makedirs("artifacts",exist_ok=True)
d=fetch_openml("heart-statlog",version=1,as_frame=True,parser="auto");X=d.data.copy();y=pd.Series(d.target).astype(str).str.lower();classes=sorted(y.unique());y=(y==classes[-1]).astype(int)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
num=X.select_dtypes(exclude=["object","category"]).columns.tolist();cat=[c for c in X.columns if c not in num]
pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)])
models={"Logistic Regression":LogisticRegression(max_iter=3000,class_weight="balanced"),"Random Forest":RandomForestClassifier(n_estimators=800,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=SEED),"HistGradientBoosting":HistGradientBoostingClassifier(max_iter=350,learning_rate=.04,l2_regularization=1,random_state=SEED)}
cv=StratifiedKFold(5,shuffle=True,random_state=SEED);out={};pipes={}
for n,m in models.items():
 p=Pipeline([("preprocess",pre),("model",m)]);s=cross_validate(p,Xtr,ytr,cv=cv,scoring=["accuracy","precision","recall","f1","roc_auc"]);out[n]={k.replace("test_",""):float(v.mean()) for k,v in s.items() if k.startswith("test_")};p.fit(Xtr,ytr);pipes[n]=p
best=max(out,key=lambda n:out[n]["roc_auc"]);p=pipes[best];pr=p.predict_proba(Xte)[:,1];pdct=(pr>=.5).astype(int)
met={"selected_model":best,"accuracy":float(accuracy_score(yte,pdct)),"precision":float(precision_score(yte,pdct)),"recall":float(recall_score(yte,pdct)),"f1":float(f1_score(yte,pdct)),"roc_auc":float(roc_auc_score(yte,pr))}
dump(p,"artifacts/disease_model.joblib");json.dump({"cv_results":out,"test_metrics":met,"columns":X.columns.tolist()},open("artifacts/metrics.json","w"),indent=2);print(json.dumps(met,indent=2))
