"""Membuat prediksi mingguan dan menambahkannya ke predictions/log.csv (hanya bertambah)."""
import glob
import os
import sys
from datetime import date

import lightgbm as lgb
import pandas as pd

from bersihkan import baca_banyak
from fitur import bangun_fitur, mingguan_lengkap

HORIZON = [4, 8]
SISTEM_MODEL = {                                   # nama sistem -> fitur yang dibuang
    "lgbm_semua": [],
    "lgbm_tanpa_kalender": ["minggu_fitri", "minggu_adha"],
}
KOLOM_LOG = ["tanggal_prediksi", "minggu_acuan", "H", "minggu_target", "sistem", "skor", "harga_acuan"]


def buat_prediksi(mingguan, f, H):
    """Kembalikan daftar (sistem, skor) untuk minggu lengkap terbaru."""
    data = f.copy()
    data["target"] = mingguan.shift(-H).reindex(data.index)
    latih = data.dropna(subset=["target"])         # hanya minggu yang jawabannya sudah diketahui
    terbaru = f.iloc[[-1]]

    hasil = []
    for nama, buang in SISTEM_MODEL.items():
        fitur = [c for c in f.columns if c not in buang]
        model = lgb.LGBMRegressor(random_state=0, verbose=-1)
        model.fit(latih[fitur], latih.target / latih.sekarang - 1)
        hasil.append((nama, float(model.predict(terbaru[fitur])[0])))
    hasil.append(("aturan_ubah4", float(terbaru.ubah4.iloc[0])))   # skor aturan sederhana
    hasil.append(("baseline_persisten", 0.0))                       # harga tidak berubah
    return hasil


def tambah_ke_log(baris, path):
    baru = pd.DataFrame(baris, columns=KOLOM_LOG)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)   # buat folder log kalau belum ada
    ada = os.path.exists(path)
    if ada:                                         # lewati baris yang sudah tercatat
        lama = pd.read_csv(path)
        sudah = set(zip(lama.minggu_acuan.astype(str), lama.H.astype(int), lama.sistem))
        belum = [(str(a), int(h), s) not in sudah
                 for a, h, s in zip(baru.minggu_acuan, baru.H, baru.sistem)]
        baru = baru[belum]
    baru.to_csv(path, mode="a", header=not ada, index=False)   # mode "a": menambah, tidak menimpa
    return len(baru)


def main(path_xlsx, path_log, hari_ini=None):
    harga = baca_banyak(path_xlsx)           # path_xlsx boleh satu file atau daftar file
    mingguan = mingguan_lengkap(harga)
    f = bangun_fitur(mingguan)
    acuan = f.index[-1]
    baris = []
    for H in HORIZON:
        for sistem, skor in buat_prediksi(mingguan, f, H):
            baris.append([str(hari_ini or date.today()), acuan.date(), H,
                          (acuan + pd.Timedelta(weeks=H)).date(), sistem, round(skor, 4),
                          round(float(mingguan.loc[acuan]), 1)])
    n = tambah_ke_log(baris, path_log)
    print(f"minggu acuan {acuan.date()}: {n} baris baru ditambahkan ke {path_log}")


if __name__ == "__main__":
    xlsx = sorted(glob.glob("data/raw/*.xlsx"))              # semua file mentah, digabung
    main(xlsx, "predictions/log.csv")
