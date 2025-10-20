# 🧠 SmartAnom: Low-Code/No-Code Anomaly Detection Framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17376652.svg)](https://doi.org/10.5281/zenodo.17376652)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

SmartAnom is a **modular, low-code/no-code anomaly detection framework** designed for researchers, engineers, and practitioners.  
It enables rapid benchmarking, hyperparameter optimization, explainability (XAI), and visualization of both classical and deep-learning models — all through an interactive graphical interface built with **Tkinter/ttk**.

---

## Key Features

- **Unified GUI Interface:**  
  Load datasets, generate synthetic data, train models, and visualize results without coding.

- **Supported Model Families:**  
  - *Isolation-Based Methods:* IF, EIF, GIF, SciForest, FairCutForest  
  - *Classical Methods:* One-Class SVM, LOF, Elliptic Envelope  
  - *Deep Models:* Autoencoder, Variational Autoencoder (VAE), DeepSVDD  

- **Novel Scoring Technique (SBAS):**  
  Implements the **Sigmoid-Based Anomaly Scoring (SBAS)** approach, which transforms path lengths via sigmoid functions and uses dynamic thresholding and majority voting to improve detection accuracy and stability.

- **Hyperparameter Optimization:**  
  Built-in Grid Search and Random Search modules.

- **Explainability (XAI):**  
  SHAP-based feature importance and model interpretability visualizations.

- **Reproducibility:**  
  Fully modular directory structure with fixed random seeds, Zenodo DOI, and exportable results (`.xlsx`, `.png`).

---

# Installation Guide

SmartAnom is designed to run smoothly on Python **3.10+**.  
Follow the detailed steps below to set up your environment and start the GUI application.

---

### 1️⃣ Clone the Repository

Open a terminal (or PowerShell on Windows) and run:

```bash
git clone https://github.com/serpil-ustebay/SmartAnom.git
cd SmartAnom

###2️⃣ Create and Activate a Virtual Environment
On Windows (PowerShell or CMD):
python -m venv smartanom
smartanom\Scripts\activate

On Linux / macOS:
python3 -m venv smartanom
source smartanom/bin/activate

##3️⃣ Upgrade pip and Install Dependencies

- Ensure you have the latest pip and setuptools:

- pip install --upgrade pip setuptools wheel


- Then install all required libraries:

- pip install -r requirements.txt


- If you want, you can manually install the essentials:

- pip install numpy pandas matplotlib scikit-learn tensorflow shap tk


- Note: On some Linux systems, you may need to install Tkinter separately:
  sudo apt-get install python3-tk

###4️⃣ Verify Installation
-Check that all modules were installed successfully:

-python -m pip list

-You should see packages such as numpy, pandas, scikit-learn, tensorflow, and shap listed.

###🧩 5️⃣ Run SmartAnom

Once everything is ready, launch the graphical interface:

python main.py



