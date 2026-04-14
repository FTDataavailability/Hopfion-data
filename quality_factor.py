# -*- coding: utf-8 -*-
"""
Quality Factor Calculation from FFT Spectrum
Author: Felipe Tejo 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Load CSV (must contain two columns: frequency, amplitude)
# ============================================================
FILE = "Quality_Factor_4096.csv"   # <-- Cambie el nombre si es necesario

df = pd.read_csv(FILE, header=None, names=["frequency", "amplitude"])
df = df.dropna().sort_values("frequency")

f = df["frequency"].to_numpy()
A = df["amplitude"].to_numpy()


# ============================================================
# Dense frequency interpolation
# ============================================================
N_dense = 50000
f_dense = np.linspace(f.min(), f.max(), N_dense)
A_dense = np.interp(f_dense, f, A)


# ============================================================
# Find peak and half-maximum
# ============================================================
imax = np.argmax(A_dense)
f0 = f_dense[imax]
A0 = A_dense[imax]
half = A0 / 2


def interp_cross(x1, y1, x2, y2, yt):
    """Linear interpolation to find the position where amplitude = yt."""
    if (y2 - y1) == 0:
        return np.nan
    return x1 + (yt - y1) * (x2 - x1) / (y2 - y1)


# ============================================================
# Left half-maximum crossing
# ============================================================
f_left = np.nan
for i in range(imax, 0, -1):
    if A_dense[i] >= half and A_dense[i - 1] < half:
        f_left = interp_cross(f_dense[i - 1], A_dense[i - 1],
                              f_dense[i], A_dense[i], half)
        break


# ============================================================
# Right half-maximum crossing
# ============================================================
f_right = np.nan
for i in range(imax, len(A_dense) - 1):
    if A_dense[i] >= half and A_dense[i + 1] < half:
        f_right = interp_cross(f_dense[i], A_dense[i],
                               f_dense[i + 1], A_dense[i + 1],
                               half)
        break


# ============================================================
# Compute FWHM and Q-factor
# ============================================================
delta_f = f_right - f_left
Q = f0 / delta_f


# ============================================================
# Print results
# ============================================================
print("\n================ RESULTS ================")
print(f"Peak frequency f0     = {f0:.6e}")
print(f"Left half-maximum     = {f_left:.6e}")
print(f"Right half-maximum    = {f_right:.6e}")
print(f"Δf (FWHM)             = {delta_f:.6e}")
print(f"Quality Factor (Q)    = {Q:.4f}")
print("=========================================\n")


# ============================================================
# Plot (optional)
# ============================================================
plt.figure(figsize=(7, 4.2))
plt.plot(f_dense, A_dense, label="Interpolated spectrum")
plt.scatter(f, A, s=10, label="Original data")

plt.axvline(f0, linestyle="--", label=f"f0 = {f0:.3e}")
plt.axhline(half, linestyle=":", label=f"A0/2 = {half:.3f}")
plt.axvline(f_left, linestyle="--", label=f"f_left = {f_left:.3e}")
plt.axvline(f_right, linestyle="--", label=f"f_right = {f_right:.3e}")

# ===== NEW: Limit the x-axis to 0 – 0.5e9 =====
plt.xlim(0, 2e9)

plt.title(f"Spectrum and FWHM (Δf = {delta_f:.3e};  Q = {Q:.3f})")
plt.xlabel("Frequency (Hz)")
plt.ylabel("FFT Magnitude")
plt.legend()
plt.tight_layout()
plt.show()

