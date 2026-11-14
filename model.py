"""Educational irrigation classifier; labels originate from simulated rules, not field observations."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / 'data' / 'synthetic_irrigation.csv'
if not DATASET.exists():
    DATASET = ROOT / 'synthetic_irrigation.csv'  # fallback untuk upload GitHub tanpa folder
FEATURES = ['soil_moisture_pct', 'field_capacity_pct', 'wilting_point_pct',
            'temperature_c', 'forecast_rain_mm', 'root_depth_cm', 'crop_stage']
CATEGORIES = ['Belum perlu', 'Pantau', 'Perlu irigasi']
STAGES = ['Vegetatif', 'Berbunga', 'Berbuah']


def validate_input(data):
    row = {key: data[key] for key in FEATURES}
    for key in FEATURES[:-1]:
        row[key] = float(row[key])
        if not np.isfinite(row[key]):
            raise ValueError(f'{key} harus berupa angka terhingga.')
    if row['crop_stage'] not in STAGES:
        raise ValueError('Fase pertumbuhan tidak dikenal.')
    if not (0 <= row['wilting_point_pct'] < row['field_capacity_pct'] <= 60):
        raise ValueError('Titik layu harus lebih kecil daripada kapasitas lapang (0–60%).')
    if not (0 <= row['soil_moisture_pct'] <= 70):
        raise ValueError('Kelembapan tanah di luar rentang 0–70%.')
    if not (10 <= row['temperature_c'] <= 45):
        raise ValueError('Suhu harus 10–45 °C.')
    if not (0 <= row['forecast_rain_mm'] <= 100):
        raise ValueError('Prakiraan hujan harus 0–100 mm.')
    if not (5 <= row['root_depth_cm'] <= 100):
        raise ValueError('Kedalaman akar harus 5–100 cm.')
    return row


def make_synthetic_dataset(n=2400, seed=42):
    rng = np.random.default_rng(seed)
    wp = rng.uniform(7, 23, n)
    fc = np.minimum(wp + rng.uniform(10, 25, n), 57)
    soil = rng.uniform(5, 60, n)
    temp = rng.uniform(20, 39, n)
    rain = np.minimum(rng.exponential(9, n), 80)
    root = rng.uniform(12, 65, n)
    stage = rng.choice(STAGES, n)
    # Synthetic teaching labels: fraction of plant-available water + simplistic stress modifiers.
    fraction = (soil - wp) / (fc - wp)
    stress = 1 - fraction + np.maximum(temp - 32, 0) * 0.018 - np.minimum(rain, 30) * 0.009
    stress += np.where(stage == 'Berbunga', 0.07, np.where(stage == 'Berbuah', 0.04, 0))
    stress += rng.normal(0, 0.035, n)
    label = np.where(stress > 0.58, 'Perlu irigasi', np.where(stress > 0.33, 'Pantau', 'Belum perlu'))
    df = pd.DataFrame(dict(soil_moisture_pct=soil, field_capacity_pct=fc,
         wilting_point_pct=wp, temperature_c=temp, forecast_rain_mm=rain,
         root_depth_cm=root, crop_stage=stage, target=label))
    for feature in FEATURES[:-1]:
        df[feature] = df[feature].round(2)
    return df


def load_dataset():
    if not DATASET.exists():
        raise FileNotFoundError('Dataset tidak ditemukan. Jalankan: python generate_data.py')
    return pd.read_csv(DATASET)


def encode_features(df):
    result = df[FEATURES].copy()
    result['crop_stage'] = result['crop_stage'].map({k: i for i,k in enumerate(STAGES)})
    if result.isna().any().any():
        raise ValueError('Ada fitur yang kosong atau kategori tidak dikenal.')
    return result


def train_model(dataset):
    x = encode_features(dataset)
    y = dataset['target']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=42,stratify=y)
    model = RandomForestClassifier(n_estimators=180, min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(x_train,y_train)
    predicted = model.predict(x_test)
    report = classification_report(y_test,predicted,output_dict=True,zero_division=0)
    return model, {'accuracy': float(accuracy_score(y_test,predicted)),
                   'macro_f1':float(report['macro avg']['f1-score']),
                   'test_n':len(y_test), 'train_n':len(y_train),
                   'confusion':confusion_matrix(y_test,predicted,labels=CATEGORIES).tolist()}


def estimate_water_deficit_mm(row, irrigation_efficiency=0.85):
    """Illustrative volumetric water-content deficit; no ET, soil hydraulics or measured calibration."""
    row=validate_input(row)
    # Percentage points * depth(cm) / 10 => mm; assumes % is volumetric water content.
    net = max(0, row['field_capacity_pct']-row['soil_moisture_pct']) * row['root_depth_cm'] / 10
    return round(net,1), round(net / irrigation_efficiency,1)


def predict_one(model, row):
    row = validate_input(row)
    prediction = model.predict(encode_features(pd.DataFrame([row])))[0]
    confidence = float(model.predict_proba(encode_features(pd.DataFrame([row]))).max())
    net, gross = estimate_water_deficit_mm(row)
    return {'status': prediction, 'model_score': round(confidence*100,1),
            'net_deficit_mm': net, 'gross_depth_mm': gross}
