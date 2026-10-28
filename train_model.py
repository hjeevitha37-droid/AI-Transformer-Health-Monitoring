from pathlib import Path
import json, joblib, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
R=Path(__file__).parent; D=pd.read_csv(R/"data"/"simulated_transformer_data.csv"); X=D.drop(columns="condition"); y=D.condition
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
models={"Random Forest":RandomForestClassifier(n_estimators=250,class_weight="balanced",random_state=42),"Gradient Boosting":GradientBoostingClassifier(random_state=42)}
rows=[]; best=None; bestscore=-1
for name,clf in models.items():
 p=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",clf)])
 p.fit(Xt,yt); pred=p.predict(Xv); score=f1_score(yv,pred,average="weighted")
 rows.append({"model":name,"accuracy":accuracy_score(yv,pred),"weighted_f1":score})
 if score>bestscore: best,bestscore,bestname=p,score,name
(R/"models").mkdir(exist_ok=True); (R/"outputs").mkdir(exist_ok=True)
joblib.dump(best,R/"models"/"transformer_condition_model.joblib")
pd.DataFrame(rows).to_csv(R/"outputs"/"model_comparison.csv",index=False)
pred=best.predict(Xv)
with open(R/"outputs"/"classification_report.json","w") as f: json.dump({"selected_model":bestname,"report":classification_report(yv,pred,output_dict=True,zero_division=0)},f,indent=2)
labels=sorted(y.unique()); sns.heatmap(confusion_matrix(yv,pred,labels=labels),annot=True,fmt="d",xticklabels=labels,yticklabels=labels)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix"); plt.tight_layout(); plt.savefig(R/"outputs"/"confusion_matrix.png",dpi=160); plt.close()
imp=pd.DataFrame({"feature":X.columns,"importance":best.named_steps["model"].feature_importances_}).sort_values("importance",ascending=False)
imp.to_csv(R/"outputs"/"feature_importance.csv",index=False)
plt.figure(figsize=(8,5)); sns.barplot(data=imp,x="importance",y="feature"); plt.tight_layout(); plt.savefig(R/"outputs"/"feature_importance.png",dpi=160); plt.close()
print("Selected:",bestname); print(pd.DataFrame(rows).to_string(index=False))
