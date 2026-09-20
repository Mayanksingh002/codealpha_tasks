import os,json
import numpy as np,pandas as pd
from joblib import dump
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.metrics import *
SEED=42; os.makedirs("artifacts",exist_ok=True)
d=fetch_openml("credit-g",version=1,as_frame=True,parser="auto")
X=d.data.copy(); y=(d.target.astype(str).str.lower()=="good").astype(int)
cat=X.select_dtypes(include=["object","category","string"]).columns.tolist()
num=[c for c in X.columns if c not in cat]
pre=ColumnTransformer([
("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
models={
"Logistic Regression":LogisticRegression(max_iter=3000,class_weight="balanced",C=1.0,random_state=SEED),
"Random Forest":RandomForestClassifier(n_estimators=800,max_features="sqrt",min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=SEED),
"HistGradientBoosting":HistGradientBoostingClassifier(max_iter=400,learning_rate=.04,max_leaf_nodes=31,l2_regularization=1,random_state=SEED)}
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
results={}; fitted={}
for name,m in models.items():
 p=Pipeline([("preprocess",pre),("model",m)])
 s=cross_validate(p,Xtr,ytr,cv=cv,scoring={"accuracy":"accuracy","precision":"precision","recall":"recall","f1":"f1","roc_auc":"roc_auc","pr_auc":"average_precision"})
 results[name]={k.replace("test_",""):float(v.mean()) for k,v in s.items() if k.startswith("test_")}
 p.fit(Xtr,ytr); fitted[name]=p
best=max(results,key=lambda n:results[n]["roc_auc"]); model=fitted[best]
prob=model.predict_proba(Xte)[:,1]; pred=(prob>=.5).astype(int)
metrics={"selected_model":best,"accuracy":float(accuracy_score(yte,pred)),"precision":float(precision_score(yte,pred)),"recall":float(recall_score(yte,pred)),"f1":float(f1_score(yte,pred)),"roc_auc":float(roc_auc_score(yte,prob)),"pr_auc":float(average_precision_score(yte,prob))}
dump(model,"artifacts/credit_model.joblib")
json.dump({"cv_results":results,"test_metrics":metrics,"columns":X.columns.tolist()},open("artifacts/metrics.json","w"),indent=2)
print(json.dumps(metrics,indent=2))
