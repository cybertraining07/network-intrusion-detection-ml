
import matplotlib.pyplot as plt
import numpy as np

models = ["Original Model", "Second-Stage Model"]

benign_alert_rates = [0.25, 10.81]
ftp_alert_rates = [0.18, 0.30]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))

bars1 = ax.bar(
    x - width / 2,
    benign_alert_rates,
    width,
    label="Benign Alert Rate"
)

bars2 = ax.bar(
    x + width / 2,
    ftp_alert_rates,
    width,
    label="FTP Alert Rate"
)

ax.set_ylabel("Alert Rate (%)")
ax.set_title("FTP Cross-Family Model Evaluation")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis="y", alpha=0.3)

ax.bar_label(bars1, fmt="%.2f%%", padding=3)
ax.bar_label(bars2, fmt="%.2f%%", padding=3)

plt.tight_layout()
plt.savefig("ftp_crossfamily_comparison.png", dpi=300)
plt.show()

print("Chart saved: ftp_crossfamily_comparison.png")
