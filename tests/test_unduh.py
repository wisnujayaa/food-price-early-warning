"""Uji pengunduh PIHPS tanpa internet: JSON contoh (bentuknya sama dengan jawaban asli PIHPS)."""
import glob
import shutil
from datetime import date

import pytest

import unduh
from bersihkan import baca_banyak, baca_pihps

CONTOH = {"data": [
    {"no": "I", "name": "Cabai Merah", "level": 1, "20/11/2026": "65,000", "23/11/2026": "63,150",
     "24/11/2026": "62,000"},
    {"no": 1, "name": "Cabai Merah Keriting ", "level": 2, "20/11/2026": "68,750", "23/11/2026": "67,500",
     "24/11/2026": "66,000"},
]}


@pytest.fixture
def folder(tmp_path):
    for f in glob.glob("data/raw/*.xlsx"):
        shutil.copy(f, tmp_path)
    return tmp_path


def test_hasil_terbaca_oleh_bersihkan(folder):
    path = unduh.main(folder=str(folder), hari_ini=date(2026, 11, 24), ambil=lambda a, b: CONTOH)
    seri = baca_pihps(path)
    assert seri.loc["2026-11-24"] == 66000.0
    assert baca_banyak(sorted(glob.glob(str(folder / "*.xlsx")))).index.max().date() == date(2026, 11, 24)


def test_idempoten(folder):
    ambil = lambda a, b: CONTOH  # noqa: E731
    assert unduh.main(folder=str(folder), hari_ini=date(2026, 11, 24), ambil=ambil) is not None
    assert unduh.main(folder=str(folder), hari_ini=date(2026, 11, 24), ambil=ambil) is None


def test_tolak_jawaban_kosong(folder):
    with pytest.raises(AssertionError):
        unduh.main(folder=str(folder), ambil=lambda a, b: {"data": [{}]})


def test_tolak_harga_tidak_wajar(folder):
    rusak = {"data": [dict(CONTOH["data"][1], **{"24/11/2026": "660"})]}    # salah ketik: 660
    with pytest.raises(AssertionError, match="rentang wajar"):
        unduh.main(folder=str(folder), hari_ini=date(2026, 11, 24), ambil=lambda a, b: rusak)


def test_tolak_format_tanggal_berubah(folder):
    rusak = {"data": [{"no": 1, "name": "Cabai Merah Keriting", "level": 2, "2026-11-24": "66,000"}]}
    with pytest.raises(ValueError):
        unduh.main(folder=str(folder), hari_ini=date(2026, 11, 24), ambil=lambda a, b: rusak)
