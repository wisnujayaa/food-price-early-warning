"""Membaca file PIHPS (xlsx) dan menghasilkan deret harga harian satu komoditas."""
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
