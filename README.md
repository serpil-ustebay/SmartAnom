# 🧠 SmartAnom: Low-Code/No-Code Anomaly Detection Framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17376652.svg)](https://doi.org/10.5281/zenodo.17376652)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

SmartAnom is a **modular, low-code/no-code anomaly detection framework** designed for researchers, engineers, and practitioners.  
It enables rapid benchmarking, hyperparameter optimization, explainability (XAI), and visualization of both classical and deep-learning models — all through an interactive graphical interface built with **Tkinter/ttk**.

---

## 🔹 Key Features

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

## ⚙️ Installation Guide

SmartAnom is designed to run smoothly on Python **3.10+**.  
Follow the detailed steps below to set up your environment and start the GUI application.

---

### 🧩 1️⃣ Clone the Repository

```bash
git clone https://github.com/serpil-ustebay/SmartAnom.git
cd SmartAnom
```

---

### 🧩 2️⃣ Create and Activate a Virtual Environment

It’s recommended to create an isolated environment named **smartanom** for this project.

#### 🔸 On Windows (PowerShell or CMD):
```bash
python -m venv smartanom
smartanom\Scripts\activate
```

#### 🔸 On Linux / macOS:
```bash
python3 -m venv smartanom
source smartanom/bin/activate
```

Once activated, you should see `(smartanom)` at the beginning of your terminal prompt:

```
(smartanom) D:\SmartAnom>
```

---

### 🧩 3️⃣ Upgrade pip and Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If you don’t have `requirements.txt` yet, manually install essentials:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow shap tk
```

💡 *Note:* On some Linux systems, you may need to install Tkinter separately:

```bash
sudo apt-get install python3-tk
```

---

### 🧩 4️⃣ Verify Installation

```bash
python -m pip list
```

---

### 🧩 5️⃣ Run SmartAnom

```bash
python main.py
```

After running this command, the **SmartAnom GUI** window will appear. You can load datasets, select models, tune parameters, and visualize results interactively.

---

### 🧩 6️⃣ (Optional) Deactivate the Environment

```bash
deactivate
```

To reactivate later:

```bash
smartanom\Scripts\activate       # Windows
# or
source smartanom/bin/activate      # Linux/macOS
```

---

## 🧠 Citation

```bibtex
@software{ustebay2025smartanom,
  author       = {Serpil Üstebay},
  title        = {SmartAnom: A Low-Code/No-Code Anomaly Detection Framework},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17376652},
  url          = {https://github.com/serpil-ustebay/SmartAnom}
}
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 💬 Contact

**Author:** Dr. Serpil Üstebay  
📧 serpil.ustebay@domain.com  
🔗 [GitHub Profile](https://github.com/serpil-ustebay)
