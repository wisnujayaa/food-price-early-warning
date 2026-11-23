"""Uji kebocoran: skor sebuah minggu tidak boleh berubah kalau masa depannya dihapus.

Untuk setiap sistem dan beberapa minggu uji t, skor dihitung dua kali:
  1. dengan deret harga lengkap
  2. dengan deret harga yang dipotong tepat di minggu t (masa depan tidak ada)
Kalau keduanya berbeda, sistem itu diam-diam memakai harga masa depan.
"""
import glob

import numpy as np
import pandas as pd
import pytest

from backtest import SISTEM, tabel
from bersihkan import baca_banyak
from fitur import bangun_fitur, mingguan_lengkap

H = 4
MINGGU_UJI = ["2021-03-07", "2022-08-14", "2023-11-26"]


@pytest.fixture(scope="module")
def mingguan():
    return mingguan_lengkap(baca_banyak(sorted(glob.glob("data/raw/*.xlsx"))))


def _skor(sistem, mingguan, t, potong):
    t = pd.Timestamp(t)
    g = tabel(mingguan, H)
    latih = g[g.index + pd.Timedelta(weeks=H) < t]               # sama seperti purging di backtest
    m = mingguan.loc[:t] if potong else mingguan
    uji = bangun_fitur(m).loc[[t]]
    return float(SISTEM[sistem](latih, uji, H, m)[0])


@pytest.mark.parametrize("sistem", list(SISTEM))
@pytest.mark.parametrize("t", MINGGU_UJI)
def test_tidak_bocor(sistem, t, mingguan):
    lengkap = _skor(sistem, mingguan, t, potong=False)
    dipotong = _skor(sistem, mingguan, t, potong=True)
    assert np.isclose(lengkap, dipotong, equal_nan=True), (
        f"{sistem} memakai masa depan di {t}: {lengkap} vs {dipotong}")


def test_uji_bisa_gagal(mingguan):
    """Tes yang tidak pernah gagal belum terbukti berguna: sistem yang sengaja bocor harus ketahuan."""
    def bocor(latih, uji, H, m):
        return (m.shift(-H).reindex(uji.index) / uji.sekarang - 1).to_numpy()   # mengintip harga t+H
    SISTEM["_bocor"] = bocor
    try:
        t = MINGGU_UJI[0]
        assert not np.isclose(_skor("_bocor", mingguan, t, False),
                              _skor("_bocor", mingguan, t, True), equal_nan=True)
    finally:
        del SISTEM["_bocor"]
