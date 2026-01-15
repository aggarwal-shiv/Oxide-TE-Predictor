Oxide TE Predictor ⚛️

Oxide TE Predictor is a machine learning framework designed to predict the thermoelectric (TE) properties of Oxide Perovskites for energy harvesting applications.

This repository contains the complete pipeline for data preprocessing, feature engineering, automated model training (with hyperparameter optimization), and a deployment-ready web application.

🔗 Live Application: www.te-predictor.com

📋 Table of Contents

Project Overview

Repository Structure

Workflow

1. Data Pre-processing

2. ML Pipeline

3. Web Application

Installation

Model Details

License

🚀 Project Overview

Thermoelectric materials can convert waste heat into electricity. This project focuses on Oxide Perovskites (ABO₃ structure) and uses ensemble machine learning techniques to predict four key properties based on chemical composition and temperature:

Electrical Conductivity ($\sigma$, S/cm)

Thermal Conductivity ($\kappa$, W/m·K)

Seebeck Coefficient ($S$, $\mu$V/K)

Figure of Merit ($zT$)

📂 Repository Structure

Oxide-TE-Predictor/
├── data/
│   ├── Dataset_new.xlsx          # Raw experimental data
│   ├── elemental_properties.xlsx # Database of atomic properties (radius, mass, etc.)
│   ├── featured_data_final.csv   # Output of pre-processing
│   └── final_data.csv            # Cleaned data ready for ML
├── final_models/                 # Directory where best trained models (.pkl) are saved
├── Data Pre-processing.ipynb     # Notebook for data cleaning & feature engineering
├── ML-pipeline.py                # Main script for training, tuning, and evaluating models
├── app.py                        # Streamlit web application script
└── README.md                     # Project documentation


🔄 Workflow

1. Data Pre-processing

File: Data Pre-processing.ipynb

This notebook handles the transformation of raw chemical formulas into machine-readable feature vectors.

Parsing: Deconstructs complex chemical formulas (e.g., La0.2Ca0.8TiO3) into elemental components.

Filtration: Filters datasets to ensure strict ABO₃ stoichiometry (1:1:3 ratio).

Feature Engineering: Calculates weighted average physical properties for A-site and B-site elements (e.g., Ionization Energy, Electronegativity, Ionic Radius) and structural descriptors like the Tolerance Factor ($t$) and Octahedral Factor ($\mu$).

2. ML Pipeline

File: ML-pipeline.py (referenced as "ML pipeline" in your description)

A robust, automated pipeline that trains multiple regressor models to find the best predictor for each target property.

Preprocessing: Handles outlier removal and drops highly correlated features ($r > 0.85$) to reduce multicollinearity.

Feature Selection: Uses Recursive Feature Elimination (RFE) to identify the most critical physical descriptors.

Models Trained: Random Forest, Gradient Boosting, AdaBoost, ExtraTrees, XGBoost, LightGBM, CatBoost, and HistGradientBoosting.

Optimization: Uses Optuna for Bayesian hyperparameter tuning (running ~2000 trials per model).

Interpretability: Generates SHAP (SHapley Additive exPlanations) plots to explain model decisions.

Output: The best-performing models are automatically serialized (pickled) into the final_models/ folder.

3. Web Application

File: app.py

A generic Streamlit interface allowing users to interact with the trained models.

Input: Users provide a chemical formula (e.g., SrTiO3) and the system parses the A/B site composition.

Real-time Calculation: The app calculates feature vectors on-the-fly using the embedded elemental_properties.xlsx data.

Visualization: Plots temperature-dependent predictions using Plotly for high-quality, interactive graphs.

Design: Features a custom, zero-gap CSS layout for a professional "dashboard" feel.

🛠️ Installation

To run this project locally, you will need Python 3.8+ and the following libraries:

pip install numpy pandas matplotlib seaborn scikit-learn scipy xgboost lightgbm catboost shap optuna streamlit plotly openpyxl


💻 Usage

Step 1: Prepare Data

Ensure your raw data is in data/Dataset_new.xlsx and run the Jupyter Notebook:

jupyter notebook "Data Pre-processing.ipynb"


Step 2: Train Models

Run the pipeline to train models and generate results in the FINAL_RESULTS directory:

python ML-pipeline.py


Step 3: Launch App

Start the web interface locally:

streamlit run app.py


🧠 Model Details

The ML pipeline evaluates models based on R² score, MAE (Mean Absolute Error), and RMSE. The final deployed models are chosen based on the highest predictive accuracy on unseen test data.

Correlation Analysis: Pearson correlation matrices are generated to ensure feature independence.

Parity Plots: Generated for every model to visualize Predicted vs. Experimental values.

📜 License

This project is open-source. Please cite the associated research paper or website (www.te-predictor.com) if you use this code or data in your work.
