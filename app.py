from pathlib import Path
import joblib, pandas as pd, streamlit as st
R=Path(__file__).parent; model_path=R/"models"/"transformer_condition_model.joblib"; data_path=R/"data"/"simulated_transformer_data.csv"
st.set_page_config(page_title="Transformer Health Monitor",page_icon="⚡",layout="wide")
st.title("⚡ AI-Based Transformer Health Condition Monitoring")
st.warning("Educational prototype using simulated data only. Do not use for real transformer safety or maintenance decisions.")
if not model_path.exists(): st.error("Model missing. Run generate_dataset.py and train_model.py first."); st.stop()
model=joblib.load(model_path)
defaults={"load_pct":65.,"ambient_temp_c":30.,"oil_temp_c":58.,"voltage_kv":11.,"oil_level_pct":90.,"moisture_ppm":25.,"vibration_mm_s":2.,"dga_h2_ppm":30.}
labels={"load_pct":"Load (%)","ambient_temp_c":"Ambient temperature (°C)","oil_temp_c":"Oil temperature (°C)","voltage_kv":"Voltage (kV)","oil_level_pct":"Oil level (%)","moisture_ppm":"Moisture (ppm)","vibration_mm_s":"Vibration (mm/s)","dga_h2_ppm":"DGA H₂ (ppm)"}
st.subheader("Transformer operating parameters"); cols=st.columns(4); values={}
for i,(k,v) in enumerate(defaults.items()):
 with cols[i%4]: values[k]=st.number_input(labels[k],value=v)
if st.button("Predict condition",type="primary"):
 row=pd.DataFrame([values]); result=model.predict(row)[0]; st.metric("Predicted category",result)
 if hasattr(model,"predict_proba"): st.caption("Class scores are not calibrated failure probabilities."); st.bar_chart(pd.DataFrame({"score":model.predict_proba(row)[0]},index=model.classes_))
 st.info("Prediction reflects synthetic training labels, not a real engineering diagnosis.")
st.subheader("Dataset preview")
if data_path.exists():
 df=pd.read_csv(data_path); st.write(f"{len(df)} simulated records"); st.dataframe(df.head(10),use_container_width=True); st.bar_chart(df.condition.value_counts())
