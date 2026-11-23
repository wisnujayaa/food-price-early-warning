"""Mengunduh harga terbaru dari PIHPS (Bank Indonesia) secara otomatis.

Dijalankan dari root repo (oleh robot GitHub setiap Sabtu pagi, atau manual):
    python src/unduh.py                  # ambil 28 hari terakhir
    python src/unduh.py --hari 60        # ambil 60 hari terakhir

Hasilnya disimpan sebagai data/raw/pihps_depok_<awal>_<akhir>.xlsx dengan tata letak yang
SAMA dengan ekspor Excel situs PIHPS, jadi bersihkan.py tidak perlu diubah. File baru hanya
dibuat kalau ada tanggal yang belum ada di data/raw/ (aman dijalankan berkali-kali).

Sumber: endpoint JSON di balik halaman "Tabel Harga > Pasar Tradisional > Berdasarkan Daerah"
(https://www.bi.go.id/hargapangan). Endpoint ini tidak didokumentasikan resmi dan bisa berubah
sewaktu-waktu; kalau berubah, skrip ini berhenti dengan pesan jelas, bukan diam-diam salah.
"""
import argparse
import glob
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bersihkan import baca_banyak  # noqa: E402

URL = "https://www.bi.go.id/hargapangan/WebSite/TabelHarga/GetGridDataDaerah"
# ID diambil dari GetRefProvince, GetRefRegency, GetRefCommodityAndCategory (dicek 2026-11-23)
PARAM = {
    "price_type_id": 1,        # 1 = pasar tradisional
    "comcat_id": "com_14",     # Cabai Merah Keriting
    "province_id": 12,         # Jawa Barat
    "regency_id": 32,          # Kota Depok
    "market_id": "",           # kosong = rata-rata semua pasar
    "tipe_laporan": 1,         # 1 = harian
}
KOMODITAS = "Cabai Merah Keriting"
HARGA_MIN, HARGA_MAKS = 5_000, 300_000    # di luar rentang ini hampir pasti salah ketik / format berubah
WIB = timezone(timedelta(hours=7))


def ambil_json(awal, akhir, timeout=60):
    """Minta data harian [awal, akhir] ke PIHPS. Mengembalikan dict JSON mentah."""
    q = dict(PARAM, start_date=awal.isoformat(), end_date=akhir.isoformat())
    req = urllib.request.Request(
        URL + "?" + urllib.parse.urlencode(q),
        headers={
            "X-Requested-With": "XMLHttpRequest",      # endpoint hanya menjawab permintaan "AJAX"
            "Accept": "application/json",
            "User-Agent": "food-price-early-warning (github.com/wisnujayaa/food-price-early-warning)",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def json_ke_tabel(js):
    """Ubah JSON PIHPS menjadi tabel bertata letak sama dengan ekspor Excel situs.

    JSON: {"data": [{"no": .., "name": .., "level": .., "DD/MM/YYYY": "67,500", ...}, ...]}
    Tabel: kolom "No", "Komoditas (Rp)", lalu satu kolom per tanggal "DD/ MM/ YYYY".
    """
    baris = [b for b in js.get("data", []) if b]              # wilayah tanpa data -> [{}]
    assert baris, "PIHPS tidak mengembalikan data sama sekali (format berubah atau wilayah kosong)"
    cocok = [b for b in baris if str(b.get("name", "")).strip() == KOMODITAS]
    assert len(cocok) == 1, f"Harus tepat 1 baris '{KOMODITAS}', ditemukan {len(cocok)}"

    tanggal = [k for k in baris[0] if k not in ("no", "name", "level")]
    for k in tanggal:
        datetime.strptime(k, "%d/%m/%Y")                       # gagal keras kalau format tanggal berubah
    tanggal.sort(key=lambda k: datetime.strptime(k, "%d/%m/%Y"))

    rapi = []
    for b in baris:
        r = {"No": b.get("no"), "Komoditas (Rp)": b.get("name")}
        for k in tanggal:
            d, m, y = k.split("/")
            r[f"{d}/ {m}/ {y}"] = b.get(k, "-")                # sama dengan ekspor situs: "05/ 10/ 2026"
        rapi.append(r)
    return pd.DataFrame(rapi)


def periksa(tabel):
    """Validasi isi sebelum disimpan. Lebih baik robot gagal daripada log terisi data rusak."""
    b = tabel[tabel["Komoditas (Rp)"].str.strip() == KOMODITAS].iloc[0, 2:]
    angka = pd.to_numeric(b.replace("-", None).astype(str).str.replace(",", ""), errors="coerce")
    ada = angka.dropna()
    assert len(ada) > 0, "Semua harga kosong"
    assert ada.between(HARGA_MIN, HARGA_MAKS).all(), (
        f"Harga di luar rentang wajar {HARGA_MIN}-{HARGA_MAKS}: {ada[~ada.between(HARGA_MIN, HARGA_MAKS)].to_dict()}")


def tanggal_terakhir_tersimpan(folder="data/raw"):
    paths = sorted(glob.glob(os.path.join(folder, "*.xlsx")))
    if not paths:
        return None
    return baca_banyak(paths).dropna().index.max().date()


def main(hari=28, folder="data/raw", hari_ini=None, ambil=ambil_json):
    akhir = hari_ini or datetime.now(WIB).date()
    awal = akhir - timedelta(days=hari)
    tabel = json_ke_tabel(ambil(awal, akhir))
    periksa(tabel)

    kolom_tgl = [datetime.strptime(c.replace(" ", ""), "%d/%m/%Y").date() for c in tabel.columns[2:]]
    terakhir_baru = max(kolom_tgl)
    terakhir_lama = tanggal_terakhir_tersimpan(folder)
    if terakhir_lama is not None and terakhir_baru <= terakhir_lama:
        print(f"Tidak ada tanggal baru (data/raw sudah sampai {terakhir_lama}).")
        return None

    path = os.path.join(folder, f"pihps_depok_{min(kolom_tgl)}_{terakhir_baru}.xlsx")
    tabel.to_excel(path, index=False)
    print(f"Disimpan {path}: {len(kolom_tgl)} hari, {min(kolom_tgl)} s.d. {terakhir_baru}.")
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hari", type=int, default=28, help="berapa hari ke belakang yang diambil")
    main(p.parse_args().hari)
