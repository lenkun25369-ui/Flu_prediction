import xgboost as xgb
import numpy as np

# 1. 載入模型
bst = xgb.Booster()
bst.load_model("xgb_model.json")

# 2. 讀取 feature 順序（來自 R）
with open("feature_order.txt", "r", encoding="utf-8") as f:
    feature_order = [line.strip() for line in f if line.strip()]

print("Feature order:", feature_order)

# 3. 範例輸入（請用你 Shiny 裡的一組真實值）
inputs = {
    "temp": 37.0,
    "height": 170,
    "weight": 65,
    "DOI": 2,
    "WOS": 1,
    "season": 1,              # 例如：1=flu season（依你原始定義）
    "rr": 18,
    "sbp": 120,
    "o2s": 98,
    "pulse": 75,
    "fluvaccine": 0,
    "cough": 0,
    "coughsputum": 0,
    "sorethroat": 0,
    "rhinorrhea": 0,
    "sinuspain": 0,
    "exposehuman": 0,
    "travel": 0,
    "medhistav": 0,
    "pastmedchronlundis": 0,
}


# 4. 依 feature_order 組成矩陣
X = np.array([[inputs[f] for f in feature_order]], dtype=np.float32)

# 5. 預測（只有 inference）
dmatrix = xgb.DMatrix(X)
prob = bst.predict(dmatrix)[0]

print(f"Predicted probability: {prob * 100:.2f}%")
