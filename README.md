# 📈 AeroFlow: End-to-End Crypto MLOps Pipeline

**AeroFlow** is a production-grade MLOps system designed to automate the complete lifecycle of a Bitcoin (BTC) price forecasting model. It solves the challenge of "Model Decay" in volatile markets by implementing an automated retraining loop triggered by data changes.

## 🛡️ Key Features
- **Data Version Control (DVC):** Large historical datasets are versioned and stored in DAGsHub remote storage, keeping the Git repository lightweight.
- **Experiment Tracking (MLflow):** Every training run logs hyperparameters, RMSE, and MAE to a centralized hosted MLflow server.
- **Automated Pipeline:** Defined in `dvc.yaml`, the pipeline handles data ingestion and model training with a single command: `dvc repro`.
- **CI/CD Automation:** GitHub Actions automatically triggers the retraining pipeline whenever new data is pushed to the repository.
- **Dual-Interface Deployment:**
    - **FastAPI (REST API):** A backend engine for machine-to-machine predictions.
    - **Streamlit (Dashboard):** An interactive visual cockpit for human analysis.

## 🛠️ Project Structure
```text
AeroFlow/
├── dashboard/           # Streamlit Frontend
├── data/                # DVC-managed Datasets
├── src/
│   ├── api/             # FastAPI Backend
│   ├── data/            # Ingestion Scripts
│   ├── model/           # Training & Prediction Logic
│   └── utils/           # MLOps Utilities
├── dvc.yaml             # Pipeline Definition
└── .github/workflows/   # CI/CD Retraining Logic
```

## 🚀 Quick Start

### 1. Setup Environment
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file from the provided `.env.example` and add your DAGsHub credentials.

### 2. Run the Pipeline
```bash
dvc repro
```

### 3. Launch the System
**Start Backend:**
```bash
python src/api/main.py
```
**Start Dashboard:**
```bash
streamlit run dashboard/app.py
```

## 📊 MLOps Dashboard
Track experiments and models live on your hosted MLflow dashboard on DAGsHub:
`https://dagshub.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.mlflow`
