"""Menilai prediksi yang sudah matang: menempelkan log prediksi dengan harga kenyataan.

Dijalankan dari root repo:  python src/evaluasi.py
"""
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from bersihkan import baca_pihps
from fitur import mingguan_lengkap

# ---------------------------------------------------------------------------
# PROTOKOL EVALUASI. Dikunci pada 2026-10-23, SEBELUM ada satu pun prediksi yang matang.
# Jangan diubah setelah melihat hasil. Kalau terpaksa diubah, catat alasannya di README.
# ---------------------------------------------------------------------------
AMBANG_LONJAKAN = 0.25      # lonjakan = harga minggu target naik lebih dari 25% dari harga acuan
MIN_MINGGU_MATANG = 26      # di bawah ini, lift tidak ditampilkan
MIN_LONJAKAN = 5            # minimal minggu lonjakan sungguhan agar lift berarti
BLOK = 8                    # panjang blok bootstrap (minggu)
ULANG = 1000                # jumlah ulangan bootstrap
BIJI = 0                    # biji acak, agar hasil bisa diulang


def tempel_kenyataan(log, mingguan):
    """Tambahkan kolom kenyataan ke log. Baris yang minggu targetnya belum ada datanya dibuang."""
    log = log.copy()
    log["minggu_target"] = pd.to_datetime(log["minggu_target"])
    log["harga_target"] = mingguan.reindex(log["minggu_target"]).to_numpy()
    matang = log.dropna(subset=["harga_target"]).copy()           # hanya yang sudah terjadi
    matang["kenaikan"] = matang["harga_target"] / matang["harga_acuan"] - 1
    matang["lonjakan"] = matang["kenaikan"] > AMBANG_LONJAKAN
    return matang


def lift_dengan_rentang(y, skor):
    """Lift = Average Precision / base rate, dengan rentang 90% dari block bootstrap."""
    y = np.asarray(y, dtype=bool)
    skor = np.asarray(skor, dtype=float)
    n = len(y)
    lift = average_precision_score(y, skor) / y.mean()
    rng = np.random.default_rng(BIJI)
    blok = min(BLOK, n)
    jumlah_blok = int(np.ceil(n / blok))
    hasil = []
    for _ in range(ULANG):
        awal = rng.integers(0, n - blok + 1, jumlah_blok)
        idx = np.concatenate([np.arange(a, a + blok) for a in awal])[:n]
        if y[idx].sum() == 0:                                      # tanpa lonjakan, lift tak terdefinisi
            continue
        hasil.append(average_precision_score(y[idx], skor[idx]) / y[idx].mean())
    lo, hi = np.percentile(hasil, [5, 95])
    return lift, lo, hi


def nilai(matang):
    """Satu baris ringkasan per (H, sistem)."""
    baris = []
    for (H, sistem), d in matang.sort_values("minggu_acuan").groupby(["H", "sistem"]):
        n, k = len(d), int(d.lonjakan.sum())
        cukup = n >= MIN_MINGGU_MATANG and k >= MIN_LONJAKAN
        if cukup:
            lift, lo, hi = lift_dengan_rentang(d.lonjakan, d.skor)
        else:
            lift = lo = hi = np.nan
        baris.append({"H": H, "sistem": sistem, "minggu_matang": n, "lonjakan": k,
                      "base_rate": round(k / n, 2), "lift": lift, "lift_5%": lo, "lift_95%": hi,
                      "status": "ok" if cukup else "belum cukup data"})
    return pd.DataFrame(baris)


def main(path_xlsx, path_log):
    mingguan = mingguan_lengkap(baca_pihps(path_xlsx))
    log = pd.read_csv(path_log)
    matang = tempel_kenyataan(log, mingguan)
    print(f"{len(log)} baris di log; {len(matang)} sudah matang.")
    if matang.empty:
        print("Belum ada prediksi yang matang.")
        return None
    hasil = nilai(matang)
    print(hasil.round(2).to_string(index=False))
    return hasil


if __name__ == "__main__":
    xlsx = sorted(glob.glob("data/raw/*.xlsx"))[-1]
    main(xlsx, "predictions/log.csv")
