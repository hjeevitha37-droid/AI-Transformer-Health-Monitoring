from pathlib import Path
import numpy as np, pandas as pd
rng=np.random.default_rng(42); n=1800
load=np.clip(rng.normal(62,20,n),5,120)
ambient=np.clip(rng.normal(30,5,n),15,48)
oil=np.clip(35+.27*load+.55*(ambient-25)+rng.normal(0,4,n),30,105)
voltage=rng.normal(11,.35,n)
level=np.clip(rng.normal(90,8,n),45,100)
moisture=np.clip(rng.gamma(2.1,12,n),1,150)
vibration=np.clip(rng.normal(2.1,.8,n),.2,8)
h2=np.clip(rng.gamma(1.8,18,n),0,250)
risk=(oil>78).astype(int)+(load>95).astype(int)+(moisture>55).astype(int)+(level<65).astype(int)+(vibration>4.5).astype(int)+(h2>90).astype(int)
noise=rng.random(n)
condition=np.where((risk>=3)|((risk>=2)&(noise>.78)),"High Risk",np.where((risk>=1)|(noise>.92),"Warning","Normal"))
df=pd.DataFrame({"load_pct":load,"ambient_temp_c":ambient,"oil_temp_c":oil,"voltage_kv":voltage,"oil_level_pct":level,"moisture_ppm":moisture,"vibration_mm_s":vibration,"dga_h2_ppm":h2,"condition":condition}).round(3)
df.to_csv(Path(__file__).parent/"data"/"simulated_transformer_data.csv",index=False)
print("Generated 1800 simulated records.")
