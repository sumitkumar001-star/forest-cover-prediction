# 🌲 Forest Cover Type Prediction

Predicting dominant tree species using ML classification models.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## 🔎 Overview

This project predicts the forest cover type (dominant tree species) from cartographic variables such as elevation, slope, soil type, and climate data. It applies machine learning classification techniques to analyze ecological patterns and assist in forest management.

---

## 🎯 Objectives

- Build a predictive model for forest cover type classification
- Compare algorithms (Decision Trees, Random Forest, Gradient Boosting)
- Evaluate performance using accuracy, precision, recall, and F1-score

---

## 📂 Project Structure

```
├── data/              # Dataset
├── templates/         # Flask HTML templates
├── saved_models/      # Trained models
├── requirements.txt   # Dependencies
├── app.py             # Flask app
└── README.md          # Documentation
```

---

## ⚙️ Installation

```bash
git clone https://github.com/sumitkumar001-star/forest-cover-prediction.git
cd forest-cover-prediction
pip install -r requirements.txt
```

---

## 📊 Dataset

**Features:** Elevation, Aspect, Slope, Soil Type, Wilderness Area

**Target:** Cover type (7 classes of tree species)

---

## 🚀 Usage

### Train the model:

```bash
python eda_and_model.py
```

### Access via browser:

```
http://127.0.0.1:5000
```

---

## 📈 Results

- **Best model:** Random Forest (~94% accuracy)
- **Key insights:** Elevation & soil type are most important predictors

---

## 🔮 Future Work

- Hyperparameter tuning
- Deployment with Streamlit
- Visualization dashboards

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License

MIT License
