🧪 Oxide TE-Predictor

Machine-Learning–Driven Prediction of Thermoelectric Properties of Oxide Perovskites

🌐 Live Web App: https://www.te-predictor.com

📌 Project Overview

Oxide TE-Predictor is an end-to-end machine-learning platform developed to predict thermoelectric (TE) properties of oxide perovskites (ABO₃) for high-temperature thermoelectric applications.

The platform integrates:

Physics-informed featurization

Advanced ensemble ML models

Hyperparameter optimization (Optuna)

Explainable AI (SHAP)

A Streamlit-based web interface embedded into a public website

The predictor simultaneously estimates the following four thermoelectric properties:

Electrical Conductivity (σ)

Thermal Conductivity (κ)

Seebeck Coefficient (S)

Figure of Merit (zT)

📁 Repository Structure
Oxide-TE-Predictor/
│
├── Data_Preprocessing.ipynb     # Complete featurization & data generation
├── ML_pipeline.py               # Training, optimization, SHAP & model export
├── app.py                       # Streamlit web application
│
├── data/
│   ├── Dataset_new.xlsx         # Raw curated experimental dataset
│   ├── elemental_properties.xlsx
│   ├── featured_data_final.csv  # Final ML-ready dataset
│   └── final_data.csv           # Cleaned & renamed dataset
│
├── final_models/
│   ├── *_σ_*.pkl
│   ├── *_κ_*.pkl
│   ├── *_S_*.pkl
│   └── *_zT_*.pkl               # Best optimized models (Feature-aware)
│
├── FINAL_RESULTS/               # Full ML outputs (per target)
│   ├── Optuna logs
│   ├── RFE results
│   ├── SHAP explanations
│   └── Parity data
│
├── Figures/
│   └── correlation_matrix.tif
│
└── README.md

🔬 Data Pre-Processing (Data_Preprocessing.ipynb)
🔹 Raw Dataset

Experimental oxide perovskite data collected manually

Stored in Dataset_new.xlsx

🔹 Key Pre-Processing Steps

Duplicate removal based on composition

Parsing chemical formulas into elemental vectors

Filtering strict ABO₃ stoichiometry

A-site, B-site, and X-site element classification

Weighted elemental property averaging

Physics-based feature engineering, including:

Goldschmidt tolerance factor (Tf)

Octahedral factor (Of)

Structural descriptors

Final feature cleanup & export

🔹 Output
data/featured_data_final.csv


This dataset is the sole input for the ML pipeline.

🤖 Machine Learning Pipeline (ML_pipeline.py)

A fully automated, reproducible, and scalable ML workflow.

🔹 Models Used

Random Forest

Extra Trees Regressor

Gradient Boosting

AdaBoost

XGBoost

LightGBM

CatBoost

Histogram Gradient Boosting

🔹 Pipeline Steps

Data loading & cleaning

Hard-range outlier removal (physics-guided)

Correlation filtering

Recursive Feature Elimination (RFECV)

Hyperparameter optimization using Optuna

5-fold cross-validated evaluation

SHAP explainability (feature importance & dependence)

Final model wrapping with feature awareness

Export of best models

🔹 Targets Predicted
Property	Symbol
Electrical Conductivity	σ
Thermal Conductivity	κ
Seebeck Coefficient	S
Figure of Merit	zT
🔹 Output Models

Saved in:

final_models/


Each model is feature-aware, ensuring consistency during deployment.

🌐 Web Application (app.py)

The Streamlit application enables real-time prediction from user-defined oxide compositions.

🔹 Features

Accepts arbitrary ABO₃ compositions (e.g. La0.2Ca0.8TiO3)

Automatic site validation (A-site / B-site)

Temperature-dependent predictions (300–1100 K)

Interactive Plotly visualizations

Physics-based feature reconstruction on-the-fly

Debug panel for transparency

🔹 Deployment

Built with Streamlit

Embedded into the public website:
👉 https://www.te-predictor.com

📊 Explainable AI (SHAP)

SHAP analysis is performed for each target:

Mean absolute SHAP importance

Feature-wise contribution

Dependence data export (no plots for scalability)

This allows physical interpretation of:

A-site vs B-site dominance

Role of ionic radii, electronegativity, and bonding

Structure–property relationships

🧠 Scientific Significance

Enables inverse materials design for oxide thermoelectrics

Identifies optimal A- and B-site chemistry

Bridges solid-state physics + machine learning

Ready for high-temperature TE material screening

📦 Requirements
python >= 3.9
numpy
pandas
scikit-learn
optuna
xgboost
lightgbm
catboost
shap
streamlit
plotly
openpyxl

📜 License

This project is intended for academic and research use.
Please cite appropriately if used in publications.

✉️ Contact

Developer: Shivam Aggarwal
Affiliation: Plasmonics & Perovskite Laboratory (PPL)
Website: https://www.te-predictor.com
