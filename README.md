# AI-ML Spacecraft Anomaly Detection

An AI-based real-time anomaly detection and autonomous decision support system for deep space missions, built as part of a course project in Artificial Intelligence & Machine Learning in Space Exploration.

## Project Overview

Deep space missions like Chandrayaan-3, Aditya-L1, and the Mars Perseverance Rover face communication delays of up to 24 minutes, making real-time human intervention during failures impossible. This project simulates an onboard AI system capable of detecting anomalies in spacecraft telemetry and recommending corrective actions autonomously.

## Dataset

**NASA CMAPSS Turbofan Engine Degradation Simulation Dataset (FD001)**  
Source: [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)  

> The dataset is not included in this repository due to size. Download it separately and place the `.txt` files in a local `CMAPSSData/` folder.

## Models Built

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Isolation Forest | 91% | 0.9357 |
| One-Class SVM | 86% | 0.9650 |
| PyTorch Autoencoder | 86% | 0.7397 |

## Autonomous Decision Layer

The system classifies engine health into five alert levels:

| Status | Condition | Action |
|---|---|---|
| NORMAL | No anomaly | Continue operation |
| WATCH | Anomaly, RUL > 100 | Increase monitoring |
| WARNING | Anomaly, RUL 51-100 | Schedule maintenance |
| CRITICAL | Anomaly, RUL 31-50 | Prepare for shutdown |
| EMERGENCY | Anomaly, RUL <= 30 | Shut subsystem / Alert ground station |

## Files

| File | Description |
|---|---|
| `Turbofan Engine Degradation Models.ipynb` | Main Jupyter notebook with all code and outputs |
| `dashboard.py` | Interactive Streamlit mission control dashboard |
| `AI & ML in Anomaly Detection.docx` | Full technical report (8-12 pages) |

## How to Run

**1. Install dependencies:**
```
pip install numpy pandas matplotlib seaborn scikit-learn plotly torch streamlit
```

**2. Run the notebook:**
Open `Turbofan Engine Degradation Models.ipynb` in Jupyter or VSCode and run all cells, making sure to read comments and placing files in the appropriate folders.

**3. Launch the dashboard:**
```
python -m streamlit run dashboard.py
```

## Tech Stack

- Python 3.14
- scikit-learn (Isolation Forest, One-Class SVM)
- PyTorch (Autoencoder)
- Streamlit + Plotly (Dashboard)
- Pandas, NumPy, Matplotlib, Seaborn
