# 🧪 Oxide TE-Predictor  
# **An Interpretable Machine Learning Framework aided Accelerated Design of Oxide Perovskites-based Thermoelectrics for Clean Energy Generation from Waste Heat**

*Shivam Aggarwal, Amandeep Saha, and Tanmoy Maiti*

*Plasmonics and Perovskites Laboratory, Department of Materials Science and Engineering, Indian Institute of Technology Kanpur, UP 208016, India.*
---
## 🔍 Project Motivation & Concept
Designing high-performance oxide thermoelectric materials is challenging due to the complex coupling between chemistry, crystal structure, and transport physics. Experimental trial-and-error approaches are time-consuming and resource intensive.

In this work, machine learning models are trained directly on experimentally reported thermoelectric data of oxide perovskites. To convert chemical compositions into meaningful numerical representations, a PAEP (Physically Aware Elemental Property) featurization strategy is employed.

The trained models learn hidden chemistry–property relationships that are difficult to capture using traditional analytical or phenomenological models.

To move beyond black-box prediction, explainable artificial intelligence techniques based on SHAP analysis are employed to extract chemically and physically meaningful trends. This enables scientific interpretation of how A-site and B-site chemistry governs thermoelectric performance.

Finally, the trained and interpreted machine learning models are deployed as a web-based application for rapid material screening and exploratory design.

The core objective of this project is not only accurate prediction, but also scientific understanding of chemistry–structure–property relationships in oxide perovskites.

---
## ✨ Key Features

- 🔬 Machine learning models trained exclusively on experimental thermoelectric data of oxide perovskites (ABO₃)
- 🔬 Physisc and chemistry-informed, PAEP (Physically Aware Elemental Property)–based featurization
- 🤖 Ensemble machine learning models (RF, Extra Trees, CatBoost, XGBoost, LightGBM)
- ⚙️ Automated hyperparameter optimization using **Optuna**
- 🧠 Explainable AI using **SHAP** for physical interpretation
- 🌡 Temperature-dependent predictions (300–1100 K)
- 🌐 Interactive **Streamlit web application**
- 📦 Ready-to-use serialized models for deployment

---

## 🎯 Predicted Thermoelectric Properties

| Property | Symbol | Unit | Physical Meaning |
|--------|--------|------|----------|
| Electrical Conductivity | σ | S·cm⁻¹ | Charge transport efficiency |
| Thermal Conductivity | κ | W·m⁻¹·K⁻¹ | Heat transport behavior |
| Seebeck Coefficient | S | µV·K⁻¹ | Thermopower and carrier type |
| Figure of Merit | zT | — | Overall thermoelectric efficiency |

Each property is modeled **independently** using optimized ML pipelines.

---

## 📁 Project Structure

```
Oxide-TE-Predictor/
│
├── Data_Preprocessing.ipynb      # Data cleaning & physics-based featurization
├── ML_pipeline.py                # Full ML pipeline (training → SHAP → export)
├── app.py                        # Streamlit web application
│
├── data/
│   ├── Dataset_new.xlsx          # Raw curated experimental dataset
│   ├── elemental_properties.xlsx # Elemental property database
│   ├── featured_data_final.csv   # Final featurized ML dataset
│   └── final_data.csv            # Cleaned & renamed dataset
│
├── final_models/                 # Optimized feature-aware ML models (.pkl)
│
├── FINAL_RESULTS/                # Optuna logs, RFE, SHAP, parity data
│
├── Figures/
│   └── correlation_matrix.tif
│
└── README.md
```

---

## 🔬 Experimental Data & PAEP Featurization (`Data_Preprocessing.ipynb`)
The experimental dataset is curated manually from reported literature and focuses exclusively on oxide perovskites with strict ABO₃ stoichiometry.
To transform raw chemical compositions into machine-learning–ready inputs, a PAEP (Physically Aware Elemental Property) featurization framework is adopted.
### Main Steps
- Manual curation of experimental oxide thermoelectric data
- Parsing chemical formulas into elemental vectors
- Strict filtering for **ABO₃ perovskite stoichiometry**
- Separation of **A-site / B-site / X-site** elements
- Weighted averaging of elemental physical properties
- Physics-based feature engineering:
  - Goldschmidt tolerance factor (Tf)
  - Octahedral factor (Of)
- Final dataset cleanup and export
  
PAEP ensures that the feature space remains physically interpretable, chemically meaningful, and consistent with solid-state principles, rather than purely statistical descriptors.

### Output Dataset
```
data/featured_data_final.csv
```

This file is the **single source of truth** for ML training.

---

## 🤖 Machine Learning Methodology (`ML_pipeline.py`)
Multiple ensemble regression models are trained independently for each thermoelectric property. Hyperparameters are optimized using Optuna, and model performance is evaluated using five-fold cross-validation.
The use of ensemble learning allows robust capture of nonlinear chemistry–property relationships while maintaining strong generalization for unseen oxide compositions.
### Models Used
- Random Forest
- Extra Trees Regressor
- Gradient Boosting
- AdaBoost
- XGBoost
- LightGBM
- CatBoost
- Histogram Gradient Boosting

### Workflow
1. Data loading & cleaning  
2. Physics-guided hard-range outlier removal  
3. Correlation-based feature filtering  
4. Recursive Feature Elimination with Cross-Validation (RFECV)  
5. Optuna-based hyperparameter optimization  
6. 5-fold cross-validation  
7. SHAP explainability analysis  
8. Feature-aware model export  

### Model Output
```
final_models/
```

---

## 🧠 Explainable AI (SHAP)
SHAP analysis is applied to the best-performing models to:
1. Identify dominant PAEP-derived chemical and structural descriptors
2. Quantify relative contributions of A-site versus B-site chemistry
3. Reveal how ionic radius, electronegativity, bonding, and structure influence thermoelectric performance
This step converts the trained machine learning models from black boxes into interpretable scientific tools and provides physics-backed insights for materials design.
All SHAP artifacts are exported for further analysis.

---

## 🌐 Web Application (`app.py`)

### Capabilities
- User-defined oxide compositions (e.g. `La0.2Ca0.8TiO3`)
- Automatic A-site and B-site validation
- Temperature-dependent predictions (300–1100 K)
- Interactive Plotly visualizations
- Physics-consistent feature reconstruction
- Debug panel for transparency

### Deployment
The Streamlit app is embedded into:

👉 https://www.te-predictor.com

---

## 🚀 Tech Stack

| Category | Technology |
|--------|-----------|
| Language | Python |
| ML Framework | scikit-learn |
| Optimization | Optuna |
| Boosting | XGBoost, LightGBM, CatBoost |
| Explainability | SHAP |
| Visualization | Plotly, Matplotlib |
| Web App | Streamlit |
| Data | Pandas, NumPy |

---

## 🛠 Installation

```bash
git clone https://github.com/yourusername/oxide-te-predictor.git
cd oxide-te-predictor
pip install -r requirements.txt
```

---

## 📜 License

This project is intended for **academic and research use**.  
Please cite appropriately if used in publications.

---

## 👤 Author & Contact

**Shivam Aggarwal**, **Amandeep Saha**, and **Tanmoy Maiti**

Plasmonics and Perovskites Laboratory, Department of Materials Science and Engineering, Indian Institute of Technology Kanpur, UP 208016, India.

🌐 https://www.te-predictor.com  

---

⭐ If you find this project useful, please consider starring the repository!
