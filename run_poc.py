# -*- coding: utf-8 -*-
"""
PoC AI Feasibility Check - PupukPintar (LKP P6, Kelompok 14)
Menguji kelayakan model regresi machine learning untuk memprediksi dosis N, P, K (kg/ha)
dari kondisi lahan, sebagaimana rencana di LKP P4 (Random Forest / Gradient Boosting).

Dataset: proxy yang dibangkitkan dari logika tabel rekomendasi resmi
(Permentan No. 40/2007 dan interpretasi kesuburan tanah Balittanah),
sesuai strategi labeling semi-automated di LKP P4: tabel resmi -> baris data tabular.
"""

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

RNG = np.random.default_rng(42)

# ----------------------------------------------------------------------
# 1) PENYUSUNAN DATA LATIH PROXY DARI LOGIKA TABEL REKOMENDASI RESMI
#    Dosis dasar mengikuti rentang Permentan 40/2007 (kg/ha):
#      Padi   : Urea 150-250, SP-36 50-100, KCl 25-75
#      Jagung : Urea 200-350, SP-36 100-200, KCl 40-100
#    Dosis disesuaikan (site-specific, pendekatan Balittanah):
#      - stok hara tanah tinggi -> dosis turun; sangat rendah -> naik
#      - target hasil naik      -> dosis naik
#      - pH masam (<5,5)        -> efisiensi P turun (P dinaikkan), perlu pengapuran
#      - tekstur liat memegang hara lebih baik -> sedikit lebih hemat
# ----------------------------------------------------------------------
TEKSTUR = {"Lempung": 1.00, "Lempung Liat": 0.97, "Liat": 0.94}
KOMODITAS = ["Padi", "Jagung"]

def status_faktor(nilai, batang):
    """Kembalikan indeks status 0..4 (Sangat Rendah..Tinggi) terhadap batang kelas Balittanah."""
    b = batang  # list 4 batas antar kelas
    if nilai < b[0]:
        return 0
    if nilai < b[1]:
        return 1
    if nilai < b[2]:
        return 2
    if nilai < b[3]:
        return 3
    return 4

KOREKSI_STATUS = [-0.20, -0.10, 0.00, 0.10, 0.20]  # koreksi proporsional terhadap stok hara

def dosis_ahli(baris):
    """Dosis rekomendasi 'paketar' (kg/ha) yang dipakai sebagai label tabel resmi."""
    f_N = status_faktor(baris.N_persen, [0.10, 0.21, 0.40, 0.60])
    f_P = status_faktor(baris.P_ppm, [5.0, 11.0, 26.0, 60.0])   # P2O5 (Bray-1)
    f_K = status_faktor(baris.K_me100g, [0.10, 0.21, 0.41, 0.80])
    kom = baris.Komoditas

    # dosis dasar pada kondisi tanah sedang dan target hasil sedang
    if kom == "Padi":
        dasar = {"N": 200.0, "P": 75.0, "K": 50.0}
        efek_target = (baris.Target_Hasil - 5.0) / 1.0   # per 1 t/ha di atas 5 t/ha
    else:
        dasar = {"N": 275.0, "P": 140.0, "K": 70.0}
        efek_target = (baris.Target_Hasil - 6.0) / 1.5   # per 1,5 t/ha di atas 6 t/ha

    dN = dasar["N"] * (1 + KOREKSI_STATUS[f_N] + 0.05 * np.clip(efek_target, -2, 3))
    dP = dasar["P"] * (1 + KOREKSI_STATUS[f_P] + 0.05 * np.clip(efek_target, -2, 3))
    dK = dasar["K"] * (1 + KOREKSI_STATUS[f_K] + 0.04 * np.clip(efek_target, -2, 3))

    # tanah masam: P terikat Al/Fe, efisiensi turun -> kebutuhan P naik sampai diapur
    if baris.pH < 5.5:
        dP *= 1.15
        dN *= 1.05
    # tekstur halus menyimpan hara lebih baik
    dN *= TEKSTUR[baris.Tekstur]
    dP *= TEKSTUR[baris.Tekstur]
    dK *= TEKSTUR[baris.Tekstur]

    # musim hujan: risiko pencucian N -> sedikit lebih tinggi, dipecah dua tahap
    if baris.Musim == "Hujan":
        dN *= 1.05

    # guardrail: kunci pada rentang aman tabel resmi
    rentang = {"Padi":  (150, 250, 50, 100, 25, 75),
               "Jagung": (200, 350, 100, 200, 40, 100)}[kom]
    dN = float(np.clip(dN, rentang[0], rentang[1]))
    dP = float(np.clip(dP, rentang[2], rentang[3]))
    dK = float(np.clip(dK, rentang[4], rentang[5]))
    return pd.Series({"Dosis_N": dN, "Dosis_P": dP, "Dosis_K": dK})

n = 6000
df = pd.DataFrame({
    "pH": RNG.uniform(3.8, 8.2, n).round(2),
    "N_persen": RNG.uniform(0.04, 1.10, n).round(3),
    "P_ppm": RNG.uniform(2, 120, n).round(1),
    "K_me100g": RNG.uniform(0.04, 1.20, n).round(3),
    "Tekstur": RNG.choice(list(TEKSTUR), n, p=[0.45, 0.35, 0.20]),
    "Komoditas": RNG.choice(KOMODITAS, n, p=[0.6, 0.4]),
    "Target_Hasil": np.nan,
    "Musim": RNG.choice(["Hujan", "Kemarau"], n, p=[0.55, 0.45]),
})
df.loc[df.Komoditas == "Padi", "Target_Hasil"] = RNG.uniform(4.0, 8.0, (df.Komoditas == "Padi").sum()).round(1)
df.loc[df.Komoditas == "Jagung", "Target_Hasil"] = RNG.uniform(4.5, 10.5, (df.Komoditas == "Jagung").sum()).round(1)

df = pd.concat([df, df.apply(dosis_ahli, axis=1)], axis=1)

FITUR = ["pH", "N_persen", "P_ppm", "K_me100g", "Tekstur", "Komoditas", "Target_Hasil", "Musim"]
X = pd.get_dummies(df[FITUR], columns=["Tekstur", "Komoditas", "Musim"], drop_first=False)
y = df[["Dosis_N", "Dosis_P", "Dosis_K"]]

# ----------------------------------------------------------------------
# 2) PELATIHAN RANDOM FOREST (5-fold cross-validation, sebagaimana target SRS)
# ----------------------------------------------------------------------
t0 = time.perf_counter()
rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)
pred = cross_val_predict(rf, X, y, cv=KFold(5, shuffle=True, random_state=42))
t_cv = time.perf_counter() - t0

rf.fit(X, y)

# ----------------------------------------------------------------------
# 3) EVALUASI: R2, MAE, dan AGREEMENT (rekomendasi dalam rentang disetujui pakar)
#    Rentang disetujui pakar pada SRS: deviasi <= 10-15% dari dosis pakar.
#    PoC memakai ambang tengah 12,5% (tegak di tengah toleransi 10-15%).
# ----------------------------------------------------------------------
AMBANG = 0.125
dev = np.abs(pred - y.values) / y.values
agreement = float((dev <= AMBANG).all(axis=1).mean() * 100)
agreement_per_hara = {
    h: round(float((dev[:, i] <= AMBANG).mean() * 100), 1) for i, h in enumerate(y.columns)
}

metrik = {
    "n_data": int(n),
    "agreement_persen": round(agreement, 1),
    "agreement_per_hara": agreement_per_hara,
    "ambang_deviasi": "12,5% (di tengah toleransi SRS 10-15%)",
    "r2_per_hara": {h: round(r2_score(y[h], pred[:, i]), 3) for i, h in enumerate(y.columns)},
    "mae_per_hara_kg": {h: round(mean_absolute_error(y[h], pred[:, i]), 1) for i, h in enumerate(y.columns)},
    "cv_waktu_detik": round(t_cv, 1),
}

# Contoh prediksi 1 lahan + keyakinan + guardrail
contoh = X.iloc[[0]]
t1 = time.perf_counter()
p1 = rf.predict(contoh)[0]
t_latensi = (time.perf_counter() - t1) * 1000

# keyakinan dari sebaran prediksi antar-pohon (perkiraan ketidakpastian)
pred_pohon = np.stack([est.predict(contoh.to_numpy())[0] for est in rf.estimators_])
cv_relatif = (pred_pohon.std(axis=0) / pred_pohon.mean(axis=0)).mean()
keyakinan = round(max(0.0, min(1.0, 1 - cv_relatif * 2)), 2)

rentang_resmi = {"Padi": (150, 250, 50, 100, 25, 75), "Jagung": (200, 350, 100, 200, 40, 100)}
lo, hi = rentang_resmi[df.Komoditas.iloc[0]][0::2], rentang_resmi[df.Komoditas.iloc[0]][1::2]
dalam_rentang = all(lo[i] - 1e-6 <= p1[i] <= hi[i] + 1e-6 for i in range(3))

metrik["contoh_prediksi"] = {
    "input": df[FITUR].iloc[0].to_dict(),
    "dosis_prediksi_kg_ha": [round(v, 1) for v in p1],
    "dosis_paketar_kg_ha": [round(v, 1) for v in y.iloc[0].values],
    "keyakinan_model": keyakinan,
    "guardrail_rentang_resmi": f"{df.Komoditas.iloc[0]}: N {lo[0]}-{hi[0]}, P {lo[1]}-{hi[1]}, K {lo[2]}-{hi[2]} kg/ha",
    "dalam_rentang_aman": bool(dalam_rentang),
    "latensi_ms": round(t_latensi, 1),
}

# Deteksi input di luar sebaran data latih (kasus meragukan -> sarankan uji lab)
def di_luar_sebaran(x):
    cek = [x.pH < 3.8 or x.pH > 8.2, x.P_ppm < 2 or x.P_ppm > 120, x.K_me100g < 0.04 or x.K_me100g > 1.20]
    return any(cek)

kasus_aneh = df.iloc[[0]].copy()
kasus_aneh["pH"] = 9.4  # tidak masuk akal utk Bogor
metrik["deteksi_luar_sebaran"] = bool(di_luar_sebaran(kasus_aneh.iloc[0]))

# ----------------------------------------------------------------------
# 4) GRAFIK RINGKAS UNTUK LAPORAN (target vs hasil, dan MAE per hara)
# ----------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
ax = axes[0]
ax.bar(["Target minimum\nSRS", "Target SRS", "Hasil PoC\n(seluruh hara)"],
       [85, 90, agreement], color=["#9fb3c8", "#9fb3c8", "#2e6f4e"], width=0.55)
ax.set_ylim(0, 100)
ax.set_ylabel("Rekomendasi dalam rentang pakar (%)")
for i, v in enumerate([85, 90, agreement]):
    ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
ax.set_title("Agreement vs target SRS", fontsize=10)

ax = axes[1]
hara = ["N", "P (P2O5)", "K (K2O)"]
nilai_r2 = [metrik["r2_per_hara"]["Dosis_N"], metrik["r2_per_hara"]["Dosis_P"], metrik["r2_per_hara"]["Dosis_K"]]
nilai_mae = [metrik["mae_per_hara_kg"]["Dosis_N"], metrik["mae_per_hara_kg"]["Dosis_P"], metrik["mae_per_hara_kg"]["Dosis_K"]]
bars = ax.bar(hara, nilai_mae, color="#b07d3c", width=0.5)
ax.set_ylabel("MAE 5-fold CV (kg/ha)")
ax.set_ylim(0, max(nilai_mae) * 1.4)
for i, (b, m, r) in enumerate(zip(bars, nilai_mae, nilai_r2)):
    ax.text(b.get_x() + b.get_width() / 2, m + 0.08, f"{m} kg/ha\nR2={r}", ha="center", fontsize=8.5)
ax.set_title("Galat prediksi per unsur hara", fontsize=10)

fig.suptitle("PoC PupukPintar: Random Forest Regressor (n=6.000, 5-fold CV)", fontsize=10.5)
fig.tight_layout()
fig.savefig("grafik_poc.png", dpi=200)

with open("hasil_poc.json", "w", encoding="utf-8") as f:
    json.dump(metrik, f, ensure_ascii=False, indent=2, default=str)
print(json.dumps({k: v for k, v in metrik.items() if k not in ("contoh_prediksi",)}, ensure_ascii=False, indent=2, default=str))
