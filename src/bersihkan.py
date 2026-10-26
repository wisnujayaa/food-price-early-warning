"""Membaca file PIHPS (xlsx) dan menghasilkan deret harga harian satu komoditas."""
import os
import pandas as pd


def baca_pihps(path_xlsx, komoditas="Cabai Merah Keriting"):
    df = pd.read_excel(path_xlsx)
    cocok = df["Komoditas (Rp)"].str.strip() == komoditas      # True/False per baris
    baris = df[cocok]
    assert len(baris) == 1, f"Harus tepat 1 baris '{komoditas}', ditemukan {len(baris)}"

    harga = baris.iloc[0, 2:]                                   # kolom tanggal saja
    harga = harga.replace("-", float("nan"))                    # "tidak ada data" -> NaN
    harga = harga.astype(str).str.replace(",", "").astype(float)    # "27,250" -> 27250.0
    harga.index = pd.to_datetime(harga.index.str.replace(" ", ""), format="%d/%m/%Y")

    assert harga.index.is_monotonic_increasing, "Tanggal tidak berurutan"
    assert harga.index.is_unique, "Ada tanggal ganda"
    harga.name = "cabai_merah_keriting"
    return harga


def baca_banyak(paths, komoditas="Cabai Merah Keriting", celah_maks_hari=5):
    """Gabungkan beberapa file PIHPS (misalnya riwayat lama + pekan terbaru) menjadi satu deret harian.

    - File diurutkan menurut tanggal akhir di nama file (pihps_depok_<awal>_<akhir>.xlsx).
    - Untuk tanggal yang ada di lebih dari satu file, nilai dari file lebih baru yang dipakai,
      kecuali nilainya kosong (maka dipakai nilai dari file lebih lama).
    - Berhenti dengan pesan jelas bila ada celah lebih dari `celah_maks_hari` hari antar tanggal
      (tanda ada pekan yang belum diunduh).
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    paths = sorted(paths, key=lambda p: os.path.basename(str(p))[-15:-5])    # tanggal akhir, format YYYY-MM-DD
    assert paths, "Tidak ada file PIHPS di data/raw/"
    gabungan = None
    for p in paths:
        seri = baca_pihps(p, komoditas)
        gabungan = seri if gabungan is None else seri.combine_first(gabungan)
    gabungan = gabungan.sort_index()
    celah = gabungan.index.to_series().diff().dropna()
    assert celah.max() <= pd.Timedelta(days=celah_maks_hari), (
        f"Ada celah {celah.max().days} hari sebelum {celah.idxmax().date()}: "
        "kemungkinan ada pekan yang belum diunduh ke data/raw/")
    gabungan.name = "cabai_merah_keriting"
    return gabungan
