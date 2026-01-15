Oxide TE Predictor ⚛️

Oxide TE Predictor is a sophisticated machine learning framework designed to predict the thermoelectric (TE) properties of Oxide Perovskites for energy harvesting applications.

This repository hosts the complete end-to-end pipeline: from raw chemical data preprocessing and engineered feature extraction to an automated ensemble ML training system and a production-ready Streamlit web application.

🔗 Live Application: www.te-predictor.com

📋 Table of Contents

Project Overview

Repository Structure

Workflow Architecture

1. Data Pre-processing

2. Automated ML Pipeline

3. Interactive Web App

Installation

Usage Guide

Model Performance & details

License

🚀 Project Overview

Thermoelectric materials are vital for sustainable energy as they convert waste heat directly into electricity. This project specifically targets Oxide Perovskites (ABO₃ structure) due to their stability and non-toxicity.

Using advanced ensemble learning, the system predicts four critical properties based solely on chemical composition and temperature:

Electrical Conductivity ($\sigma$, S/cm)

Thermal Conductivity ($\kappa$, W/m·K)

Seebeck Coefficient ($S$, $\mu$V/K)

Figure of Merit ($zT$)

📂 Repository Structure

Oxide-TE-Predictor/
├── data/
│   ├── Dataset_new.xlsx          # Original experimental dataset
│   ├── elemental_properties.xlsx # Atomic properties database (Radius, Ionization energy, etc.)
│   ├── featured_data_final.csv   # Intermediate output with engineered features
│   └── final_data.csv            # Cleaned, highly-correlated features removed
├── final_models/                 # Production-ready pickled models (CatBoost, ExtraTrees, etc.)
├── Figures/                      # Generated plots (Correlation matrices, SHAP plots)
├── Data Pre-processing.ipynb     # Jupyter notebook for data cleaning & vectorization
├── ML-pipeline.py                # The AutoML script (Training, RFE, Optuna, Evaluation)
├── app.py                        # Streamlit dashboard source code
└── README.md                     # Documentation


🔄 Workflow Architecture

The project follows a strict three-stage pipeline to ensure data integrity and model robustness.

1. Data Pre-processing

Source: Data Pre-processing.ipynb

This stage transforms raw chemical strings into mathematical vectors suitable for machine learning.

Stoichiometry Filtering: The script strictly enforces the perovskite structure (ABO₃) by checking elemental ratios (1:1:3).

Elemental Vectorization: Deconstructs formulas (e.g., La0.2Ca0.8TiO3) and maps them to A-site and B-site components.

Property Weighting: Calculates weighted average physical properties for each site using elemental_properties.xlsx.

Features calculated: Ionic Radius, Electronegativity, Ionization Energy, Atomic Mass, Melting Point, etc.

Structural Factors: Computes domain-specific engineered features:

Tolerance Factor ($t$): Determines structural stability.

Octahedral Factor ($\mu$): Ratio of B-site radius to X-site radius.

2. Automated ML Pipeline

Source: ML-pipeline.py

A fully automated training script that acts as an AutoML system specialized for materials science.

Feature Selection: * Removes highly correlated features (Pearson $r > 0.85$) to prevent multicollinearity.

Uses Recursive Feature Elimination (RFE) with Cross-Validation to select the most impactful descriptors.

Ensemble Training: Trains 8 distinct regression algorithms simultaneously:

Tree-based: Random Forest, ExtraTrees

Boosting: XGBoost, LightGBM, CatBoost, Gradient Boosting, AdaBoost, HistGradientBoosting

Hyperparameter Optimization: Utilizes Optuna to run ~2000 trials per model, optimizing parameters (learning rate, depth, regularization) via Bayesian search.

Explainability: Integrates SHAP (SHapley Additive exPlanations) to generate feature importance and dependence plots.

Artifact Generation: Automatically saves the best-performing model for each target property into the final_models/ directory.

3. Interactive Web App

Source: app.py

A user-facing dashboard built with Streamlit that allows researchers to screen new materials instantly.

On-the-fly Calculation: The app does not look up pre-calculated values. It parses the user's input formula, calculates the structural features in real-time, and feeds them into the loaded ML models.

Zero-Gap Layout: Custom CSS implementation for a professional, compact scientific tool interface.

Interactive Visualization: Uses Plotly to render high-resolution, interactive temperature-dependent property curves.

🛠️ Installation

To run this project locally, ensure you have Python 3.8 or higher installed.

Clone the repository:

git clone [https://github.com/yourusername/Oxide-TE-Predictor.git](https://github.com/yourusername/Oxide-TE-Predictor.git)
cd Oxide-TE-Predictor


Install dependencies:

pip install numpy pandas matplotlib seaborn scikit-learn scipy xgboost lightgbm catboost shap optuna streamlit plotly openpyxl


💻 Usage Guide

1. Generating Training Data

If you have new raw data in Dataset_new.xlsx, run the preprocessing notebook to generate the feature vectors:

jupyter notebook "Data Pre-processing.ipynb"


2. Training the Models

To retrain the models (this may take several hours due to Optuna optimization):

python ML-pipeline.py


Check the logs in the console for real-time training progress and R² scores.

3. Running the Web App

To launch the prediction interface locally:

streamlit run app.py


🧠 Model Performance & Details

The pipeline rigorously evaluates models using 5-fold Cross-Validation. The final deployed models were selected based on the highest R² Score and lowest RMSE on the test set.

Current Production Models:

Seebeck Coefficient ($S$): ExtraTrees Regressor

Electrical Conductivity ($\sigma$): CatBoost Regressor

Thermal Conductivity ($\kappa$): Gradient Boosting Regressor

Figure of Merit ($zT$): CatBoost Regressor

Detailed parity plots and correlation matrices can be found in the Figures/ directory after running the pipeline.

📜 License

This project is open-source. If you use this code or the pre-trained models in your research, please cite:

Oxide TE Predictor: A Machine Learning Framework for Perovskite Thermoelectrics > Available at: www.te-predictor.com
