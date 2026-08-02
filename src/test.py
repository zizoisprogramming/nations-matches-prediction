

import pandas as pd
import numpy as np
import os 
from pathlib import Path
import datetime as dt

from sklearn.metrics import classification_report
from src.feature_extraction import FeatureExtraction
from src.feature_scaling import FeatureScaling
from src.feature_selection import FeatureSelection
from src.inference import predict_proba, predict


path = "/Users/ziadsamer/Projects/nations-matches-prediction/data/test/test.csv"
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

# results = predict_proba(path)
# print(results.to_string(index=False))
df = pd.read_csv("/Users/ziadsamer/Projects/nations-matches-prediction/data/test/test.csv")[:5]
y_true = np.select(
    [
        df['home_score'] > df['away_score'],
        df['home_score'] < df['away_score']
    ],
    [
        1,  
        2   
    ],
    default=0 
)

y_pred = predict(path=path)
print(classification_report(y_true, y_pred))