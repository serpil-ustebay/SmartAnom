"""
get_param_grid.py
-----------------
Central configuration for all model hyperparameters in SmartAnom.
Each model has:
    - defaults: used for Configure Hyperparameters in the main GUI
    - grid: used for Hyperparameter Search optimization

Author: Serpil Üstebay
"""

def get_param_grid_for_model(model_name: str):
    grids = {
        # 🌲 Isolation Forest
        "Isolation Forest": {
            "defaults": {
                "n_trees": 100,
                "sample_size": 256,
                "contamination": 0.1,
                "threshold": 0.95,
                "majority": 5
            },
            "grid": {
                "n_trees": [50, 100],
                "sample_size": [128, 256],
                "contamination": [0.05, 0.1],
                "threshold": [0.95, 0.98],
                "majority": [7, 10]
            }
        },

        # 🌳 Extended Isolation Forest
        "Extended Isolation Forest": {
            "defaults": {
                "n_trees": 100,
                "sample_size": 256,
                "level": 1,
                "contamination": 0.1,
                "threshold": 0.95,
                "majority": 5
            },
            "grid": {
                "n_trees": [50, 100],
                "sample_size": [128, 256],
                "contamination": [0.05, 0.1],
                "threshold": [0.95, 0.98],
                "majority": [7, 10]
            }
        },

        # 🔷 Generalized Isolation Forest
        "Generalized Isolation Forest": {
            "defaults": {
                "n_trees": 100,
                "sample_size": 256,
                "k_planes": 2,
                "contamination": 0.1,
                "threshold": 0.95,
                "majority": 5
            },
            "grid": {
                "n_trees": [50, 100],
                "sample_size": [128, 256],
                "contamination": [0.05, 0.1],
                "threshold": [0.95, 0.98],
                "majority": [7, 10]
            }
        },

        # 🧩 SciForest
        "SciForest": {
            "defaults": {
                "n_trees": 100,
                "sample_size": 256,
                "contamination": 0.1,
                "threshold": 0.95,
                "majority": 5
            },
            "grid": {
                "n_trees": [50, 100],
                "sample_size": [128, 256],
                "contamination": [0.05, 0.1],
                "threshold": [0.95, 0.98],
                "majority": [7, 10]
            }
        },

        # ⚖️ FairCutForest
        "FairCutForest": {
            "defaults": {
                "n_trees": 100,
                "sample_size": 256,
                "contamination": 0.1,
                "threshold": 0.95,
                "majority": 5
            },
            "grid": {
                "n_trees": [50, 100],
                "sample_size": [128, 256],
                "contamination": [0.05, 0.1],
                "threshold": [0.95, 0.98],
                "majority": [7, 10]
            }
        },

        # 🧠 One-Class SVM
        "One-Class SVM": {
            "defaults": {
                "kernel": "rbf",
                "nu": 0.05,
                "gamma": "scale"
            },
            "grid": {
                "kernel": ["rbf"],
                "nu": [0.01, 0.05],
                "gamma": ["scale"]
            }
        },

        # 🔍 Local Outlier Factor
        "Local Outlier Factor": {
            "defaults": {
                "n_neighbors": 20,
                "leaf_size": 30,
                "metric": "minkowski",
                "contamination": 0.1
            },
            "grid": {
                "n_neighbors": [10, 20, 35],
                "leaf_size": [30, 50, 70],
                "metric": ["minkowski", "euclidean"],
                "contamination": [0.05, 0.1, 0.15]
            }
        },

        # 📈 Elliptic Envelope
        "Elliptic Envelope": {
            "defaults": {
                "contamination": 0.1,
                "support_fraction": 0.8,
                "assume_centered": False
            },
            "grid": {
                "contamination": [0.05, 0.1, 0.15],
                "support_fraction": [0.7, 0.8, 0.9],
                "assume_centered": [False, True]
            }
        },

        # 🤖 Autoencoder
        "Autoencoder": {
            "defaults": {
                "latent_dim": 8,
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 50,
                "dropout": 0.2
            },
            "grid": {
                "latent_dim": [4, 8, 16],
                "learning_rate": [1e-3, 1e-4],
                "batch_size": [32, 64, 128],
                "epochs": [50, 100],
                "dropout": [0.0, 0.2, 0.5]
            }
        },

        # 🧬 Variational Autoencoder (VAE)
        "VAE": {
            "defaults": {
                "latent_dim": 8,
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 50,
                "beta": 1.0
            },
            "grid": {
                "latent_dim": [4, 8, 16],
                "learning_rate": [1e-3, 1e-4],
                "batch_size": [32, 64, 128],
                "epochs": [50, 100],
                "beta": [0.5, 1.0, 2.0]
            }
        },

        # 🧠 DeepSVDD
        "DeepSVDD": {
            "defaults": {
                "latent_dim": 16,
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 50,
                "lambda_reg": 0.5
            },
            "grid": {
                "latent_dim": [8, 16, 32],
                "learning_rate": [1e-3, 1e-4],
                "batch_size": [32, 64, 128],
                "epochs": [50, 100],
                "lambda_reg": [0.1, 0.5, 1.0]
            }
        },
    }

    return grids.get(model_name, {})
