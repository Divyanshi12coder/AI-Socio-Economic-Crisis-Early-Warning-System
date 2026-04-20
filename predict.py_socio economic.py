import joblib
import numpy as np

def predict_crisis(input_data):
    model = joblib.load("models/crisis_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)
    return "Crisis Likely" if prediction[0] == 1 else "Stable"

if __name__ == "__main__":
    sample = np.array([3.2, 7.5, 1200, 5.6])  # Example socio-economic indicators
    print(predict_crisis(sample))
