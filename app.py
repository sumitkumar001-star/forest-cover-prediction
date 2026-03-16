import numpy as np
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

try:
    model = joblib.load('saved_models/model.joblib')
    scaler = joblib.load('saved_models/scaler.joblib')
except FileNotFoundError:
    model = None
    scaler = None

cover_type_mapping = {
    1: 'Spruce/Fir', 2: 'Lodgepole Pine', 3: 'Ponderosa Pine',
    4: 'Cottonwood/Willow', 5: 'Aspen', 6: 'Douglas-fir', 7: 'Krummholz'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return render_template('result.html', prediction_text="Error: Model not loaded. Run eda_and_model.py first.")

    try:
        numerical_inputs = [float(request.form.get(f, 0)) for f in [
            'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
            'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
        ]]
        wilderness_area = int(request.form.get('Wilderness_Area', 1))
        soil_type = int(request.form.get('Soil_Type', 1))

        numerical_features_scaled = scaler.transform([numerical_inputs])
        wilderness_features = np.zeros(4)
        if 1 <= wilderness_area <= 4:
            wilderness_features[wilderness_area - 1] = 1
        soil_features = np.zeros(40)
        if 1 <= soil_type <= 40:
            soil_features[soil_type - 1] = 1

        final_features = np.concatenate([numerical_features_scaled.flatten(), wilderness_features, soil_features]).reshape(1, -1)
        
        prediction_val = model.predict(final_features)
        predicted_cover_type_name = cover_type_mapping.get(prediction_val[0], "Unknown")
        
        result_text = f'Predicted Cover Type: {predicted_cover_type_name}'
        return render_template('result.html', prediction_text=result_text)
    except Exception as e:
        return render_template('result.html', prediction_text=f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(debug=True)

