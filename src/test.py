

import pandas as pd
import os 
from pathlib import Path
import datetime as dt

from src.feature_extraction import FeatureExtraction
from src.feature_scaling import FeatureScaling
from src.feature_selection import FeatureSelection
from src.inference import predict_proba


path = "/Users/ziadsamer/Projects/nations-matches-prediction/data/test/sec_test.csv"
BASE_DIR = Path(__file__).parent.parent
save_dir = BASE_DIR / "data" / str(dt.date.today())
try:
    os.makedirs(save_dir, exist_ok=True)
except:
    raise Exception("alao")

fe = FeatureExtraction()
path = fe.run(path, save_dir)

fsc = FeatureScaling()
path = fsc.run(path, save_dir)

fsl = FeatureSelection()
path = fsl.run(path, save_dir)

results = predict_proba(path)
print(results.to_string(index=False))
