import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Veriyi DataFrame olarak tanımla
data = {
    "Dataset": ["Blobs","Blobs","Blobs","Blobs","Blobs",
                "Circles","Circles","Circles","Circles","Circles",
                "Helix","Helix","Helix","Helix","Helix",
                "Moons","Moons","Moons","Moons","Moons",
                "Sinusoidal","Sinusoidal","Sinusoidal","Sinusoidal","Sinusoidal",
                "Spiral","Spiral","Spiral","Spiral","Spiral"],
    "Algorithm": ["EIF","FairCutForest","GIF","SCiForest","IF"]*6,
    "Accuracy_SBAS": [0.9926,0.9930,0.9917,0.9913,0.9917,
                      0.9901,0.9901,0.9901,0.9901,0.9901,
                      0.9950,0.9950,0.9952,0.9950,0.9977,
                      0.9913,0.9901,0.9909,0.9909,0.9913,
                      0.9901,0.9901,0.9901,0.9901,0.9901,
                      0.9901,0.9901,0.9901,0.9901,0.9901],
    "Accuracy_MBAS": [0.9880,0.9266,0.9802,0.9777,0.9757,
                      0.9901,0.9546,0.9901,0.9889,0.9876,
                      0.9950,0.9932,0.9925,0.9948,0.9838,
                      0.9880,0.9443,0.9814,0.9831,0.9790,
                      0.9864,0.9715,0.9806,0.9835,0.9777,
                      0.9893,0.8890,0.9810,0.9901,0.9600]
}
df = pd.DataFrame(data)

datasets = ["Blobs", "Circles", "Helix", "Moons", "Sinusoidal", "Spiral"]

# Radar plot fonksiyonu
def plot_radar(ax, algo_data, algo_name):
    N = len(datasets)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # SBAS
    values = algo_data["Accuracy_SBAS"].tolist()
    values += values[:1]
    ax.plot(angles, values, color="blue", linewidth=2, label="SBAS")
    ax.fill(angles, values, color="blue", alpha=0.25)

    # MBAS
    values = algo_data["Accuracy_MBAS"].tolist()
    values += values[:1]
    ax.plot(angles, values, color="orange", linewidth=2, label="MBAS")
    ax.fill(angles, values, color="orange", alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(datasets)
    ax.set_title(algo_name, size=12, y=1.1)
    ax.set_ylim(0.88, 1.0)

# Algoritma sırası
algo_order = ["IF", "EIF", "GIF", "SCiForest", "FairCutForest"]

# --- Çizim ---
fig = plt.figure(figsize=(16, 10))

# Üst satır: 3 sütun
gs_top = fig.add_gridspec(1, 3, top=0.95, bottom=0.55)
for i, algo in enumerate(algo_order[:3]):
    ax = fig.add_subplot(gs_top[0, i], polar=True)
    algo_data = df[df["Algorithm"] == algo]
    plot_radar(ax, algo_data, algo)

# Alt satır: 2 sütun (ortalanmış)
gs_bottom = fig.add_gridspec(1, 2, top=0.45, bottom=0.05)
for i, algo in enumerate(algo_order[3:]):
    ax = fig.add_subplot(gs_bottom[0, i], polar=True)
    algo_data = df[df["Algorithm"] == algo]
    plot_radar(ax, algo_data, algo)

# Legend
plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1.05))
plt.tight_layout()
plt.savefig("rdrSent.pdf")
plt.show()
