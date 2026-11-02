"""Arena uji: menilai SEMUA sistem dengan cara yang identik.

Dijalankan dari root repo:
    python src/backtest.py                      # periode dev (2020-2023), semua sistem
    python src/backtest.py --sistem lgbm_semua aturan_ubah4 --catatan "uji awal"

Setiap jalan dicatat ke experiments.csv (termasuk yang hasilnya jelek), supaya kita
tahu sudah berapa kali mencoba. Periode ujian akhir sengaja dikunci (lihat PERIODE).
"""
import argparse
import glob
import os
import subprocess
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from bersihkan import baca_banyak
from fitur import bangun_fitur, mingguan_lengkap

# ---------------------------------------------------------------------------
# PEMBAGIAN DATA. Dikunci 2026-11-02.
#   dev   : boleh dipakai berkali-kali untuk mencoba ide.
#   ujian : dibuka SEKALI di Modul 8, hanya untuk kandidat yang sudah dikunci.
# Setiap tahun uji dilatih hanya dengan data sebelumnya (rolling origin + purging).
# ---------------------------------------------------------------------------
PERIODE = {
    "dev":   [(f"{t}-01-01", f"{t + 1}-01-01") for t in range(2020, 2024)],
    "ujian": [("2024-01-01", "2025-01-01"), ("2025-01-01", "2026-01-01"), ("2026-01-01", "2026-11-01")],
}
AMBANG_LONJAKAN = 0.25   # sama dengan protokol log prospektif (src/evaluasi.py)
BLOK, ULANG, BIJI = 8, 1000, 0
PATH_EKSPERIMEN = "experiments.csv"


# ---------------------------------------------------------------------------
# SISTEM. Satu sistem = fungsi(latih, uji, H) -> skor untuk setiap baris uji.
# Skor makin besar = makin berisiko melonjak. Untuk menambah ide baru, tulis fungsi
# baru dan daftarkan di SISTEM. Jangan ubah sistem lama: buat nama baru.
# ---------------------------------------------------------------------------
def _lgbm(buang=()):
    def skor(latih, uji, H):
        fitur = [c for c in uji.columns if c not in ("target", *buang)]
        model = lgb.LGBMRegressor(random_state=0, verbose=-1)
        model.fit(latih[fitur], latih.target / latih.sekarang - 1)
        return model.predict(uji[fitur])
    return skor


SISTEM = {
    "lgbm_semua":          _lgbm(),
    "lgbm_tanpa_kalender": _lgbm(buang=("minggu_fitri", "minggu_adha")),
    "aturan_ubah4":        lambda latih, uji, H: uji.ubah4.to_numpy(),
    "baseline_persisten":  lambda latih, uji, H: np.zeros(len(uji)),
}


def tabel(mingguan, H, bangun=bangun_fitur):
    """Tabel fitur + kolom target (harga H minggu lagi)."""
    f = bangun(mingguan)
    f["target"] = mingguan.shift(-H).reindex(f.index)
    return f.dropna(subset=["target"])


def jalankan(mingguan, H, nama_sistem, periode="dev"):
    """Rolling origin: untuk tiap potongan uji, latih hanya dengan data yang jawabannya
    sudah diketahui sebelum potongan itu dimulai (purging)."""
    g = tabel(mingguan, H)
    potongan = []
    for awal, akhir in PERIODE[periode]:
        awal, akhir = pd.Timestamp(awal), pd.Timestamp(akhir)
        uji = g[(g.index >= awal) & (g.index < akhir)].copy()
        latih = g[g.index + pd.Timedelta(weeks=H) < awal]       # target latih jatuh sebelum uji
        # setiap sistem melihat salinan bersih: skor sistem lain tidak boleh jadi fitur
        skor = {nama: SISTEM[nama](latih, uji.copy(), H) for nama in nama_sistem}
        potongan.append(uji.assign(**skor))
    hasil = pd.concat(potongan)
    hasil["lonjakan"] = hasil.target / hasil.sekarang - 1 > AMBANG_LONJAKAN
    return hasil


def _indeks_blok(n, rng):
    blok = min(BLOK, n)
    awal = rng.integers(0, n - blok + 1, int(np.ceil(n / blok)))
    return np.concatenate([np.arange(a, a + blok) for a in awal])[:n]


def lift(y, skor):
    return average_precision_score(y, skor) / np.mean(y)


def nilai(hasil, nama_sistem, pembanding="aturan_ubah4"):
    """Lift tiap sistem + rentang 90%, dan selisih lift terhadap pembanding (paired bootstrap:
    blok minggu yang SAMA dipakai untuk kedua sistem, jadi noise-nya saling meniadakan)."""
    y = hasil.lonjakan.to_numpy()
    S = {n: hasil[n].to_numpy() for n in nama_sistem}
    rng = np.random.default_rng(BIJI)
    sampel = {n: [] for n in nama_sistem}
    selisih = {n: [] for n in nama_sistem}
    for _ in range(ULANG):
        idx = _indeks_blok(len(y), rng)
        if y[idx].sum() == 0:
            continue
        l = {n: lift(y[idx], S[n][idx]) for n in nama_sistem}
        for n in nama_sistem:
            sampel[n].append(l[n])
            if pembanding in l:
                selisih[n].append(l[n] - l[pembanding])
    baris = []
    for n in nama_sistem:
        lo, hi = np.percentile(sampel[n], [5, 95])
        b = {"sistem": n, "lift": lift(y, S[n]), "lift_5%": lo, "lift_95%": hi}
        if pembanding in S and n != pembanding:
            b["vs_" + pembanding + "_5%"], b["vs_" + pembanding + "_95%"] = np.percentile(selisih[n], [5, 95])
        baris.append(b)
    return pd.DataFrame(baris)


def catat(df, catatan, path=PATH_EKSPERIMEN):
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        kotor = subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True,
                               text=True).stdout.strip()
        if kotor:
            commit += "+belum_commit"     # kode yang dipakai belum tersimpan di git
    except OSError:
        commit = ""
    df = df.copy()
    df.insert(0, "waktu", datetime.now().isoformat(timespec="seconds"))
    df.insert(1, "commit", commit)
    df["catatan"] = catatan
    df.round(4).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def main(paths, nama_sistem, periode="dev", catatan="", path_eksperimen=PATH_EKSPERIMEN):
    if periode == "ujian":
        raise SystemExit("Periode ujian dikunci sampai Modul 8. Jangan dibuka untuk mencoba ide.")
    mingguan = mingguan_lengkap(baca_banyak(paths))
    semua = []
    for H in (4, 8):
        hasil = jalankan(mingguan, H, nama_sistem, periode)
        r = nilai(hasil, nama_sistem)
        r.insert(0, "H", H)
        r.insert(1, "periode", periode)
        r.insert(2, "minggu", len(hasil))
        r.insert(3, "lonjakan", int(hasil.lonjakan.sum()))
        semua.append(r)
    df = pd.concat(semua, ignore_index=True)
    print(df.round(2).to_string(index=False))
    if path_eksperimen:
        catat(df, catatan, path_eksperimen)
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sistem", nargs="+", default=list(SISTEM))
    p.add_argument("--periode", default="dev", choices=list(PERIODE))
    p.add_argument("--catatan", default="")
    a = p.parse_args()
    main(sorted(glob.glob("data/raw/*.xlsx")), a.sistem, a.periode, a.catatan)
