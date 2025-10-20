import numpy as np
import matplotlib.pyplot as plt

class SigmoidUtils:
    @staticmethod
    def custom_sigmoid(x, k=2):
        return 1 / (1 + np.exp(-k * x))

x_values = np.arange(-10, 11)  # -10'dan 10'a
k_values = [0.5, 1, 2, 5, 10]

plt.figure(figsize=(10,6))
colors = ['blue', 'orange', 'green', 'red', 'purple']

for i, k in enumerate(k_values):
    y_values = SigmoidUtils.custom_sigmoid(x_values, k=k)
    plt.plot(x_values, y_values, color=colors[i], linewidth=2, alpha=0.8, label=f'k={k}')

plt.xticks(np.arange(-10, 11, 1))
plt.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.7)  # legend kaldırıldı
plt.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
#plt.title("Sigmoid Transformation with Different k", fontsize=14, weight='bold')
plt.xlabel("x", fontsize=12)
plt.ylabel("Sigmoid(x)", fontsize=12)
plt.legend(title="Steepness (k)", fontsize=10, title_fontsize=11, loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('SigmoidPlot.pdf')
plt.show()
