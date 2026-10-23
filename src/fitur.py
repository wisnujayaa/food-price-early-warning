"""Membuat tabel fitur mingguan dari deret harga harian."""
import math
import pandas as pd

# Tanggal hari raya (belum semuanya diverifikasi terhadap keputusan resmi pemerintah)
FITRI = pd.to_datetime(["2017-06-25", "2018-06-15", "2019-06-05", "2020-05-24", "2021-05-13",
                        "2022-05-02", "2023-04-22", "2024-04-10", "2025-03-31", "2026-03-21",
                        "2027-03-10"])
ADHA = pd.to_datetime(["2017-09-01", "2018-08-22", "2019-08-11", "2020-07-31", "2021-07-20",
                       "2022-07-10", "2023-06-29", "2024-06-17", "2025-06-06", "2026-05-27",
                       "2027-05-17"])


def minggu_ke(hari_raya, t):
    """Berapa minggu lagi sampai hari raya berikutnya (dihitung dari tanggal t)."""
    berikut = hari_raya[hari_raya >= t].min()
    assert pd.notna(berikut), f"Daftar hari raya habis sebelum {t.date()}, tambahkan tanggal baru"
    return math.ceil((berikut - t).days / 7)


def mingguan_lengkap(harga):
    """Rata-rata harga per minggu (Senin-Minggu). Minggu terakhir dibuang jika belum lengkap."""
    mingguan = harga.resample("W-SUN").mean().ffill()
    terakhir_ada_data = harga.dropna().index.max()
    penutup_terakhir = mingguan.index[-1]                       # hari Minggu penutup minggu terakhir
    # data PIHPS hanya hari kerja: minggu dianggap lengkap bila data sudah sampai hari Jumat
    if terakhir_ada_data < penutup_terakhir - pd.Timedelta(days=2):
        mingguan = mingguan.iloc[:-1]
    return mingguan


def bangun_fitur(mingguan):
    f = pd.DataFrame({"sekarang": mingguan})
    f["lag1"]   = mingguan.shift(1)
    f["lag4"]   = mingguan.shift(4)
    f["ubah1"]  = mingguan / mingguan.shift(1) - 1
    f["ubah4"]  = mingguan / mingguan.shift(4) - 1
    f["rata4"]  = mingguan.rolling(4).mean()
    f["rata52"] = mingguan.rolling(52).mean()
    f = f.dropna()
    f["minggu_fitri"] = [minggu_ke(FITRI, t) for t in f.index]
    f["minggu_adha"]  = [minggu_ke(ADHA, t) for t in f.index]
    return f
