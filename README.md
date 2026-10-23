# food-price-early-warning

An honest attempt to build an early-warning signal for chili-price spikes in Indonesia, with a pre-registered, append-only log of live predictions.

**Status (Oct 2026): the signal is weak.** On 2020–2024 backtests, a gradient-boosting model ranks spike weeks only ~1.4× better than chance, and it does *not* beat the naive "price stays the same" forecast on average error. This repo exists to measure that carefully and to test it forward in time, not to claim a working forecaster.

*Ringkasan singkat dalam Bahasa Indonesia ada di bagian paling bawah.*

## Question

Can we warn, 4 or 8 weeks ahead, that the price of **Cabai Merah Keriting (curly red chili) in Kota Depok** is about to jump?

- Data: PIHPS (Bank Indonesia's national strategic food price information centre), daily prices, weekdays only, from 2017-08-16.
- Unit of analysis: weekly mean price (week = Monday to Sunday).
- Horizons: H = 4 and H = 8 weeks.

## Method (what makes the numbers trustworthy)

- **Rolling-origin validation.** Test years 2020, 2021, 2022, 2023, 2024. For each test year the model is trained only on earlier data.
- **Purging.** Training rows whose target week reaches into the test period are removed (`index + H weeks < start of test year`), so answers never leak from test into training.
- **Baselines first.** (A) a simple rule, the price change over the last 4 weeks; (B) persistence, "next price = current price".
- **Spike metrics, not accuracy.** A spike is rare (base rate about 18% of weeks), so plain accuracy is misleading. I use precision, recall, the precision-recall curve and **lift = Average Precision ÷ base rate** (1.0 means no better than chance).
- **Uncertainty everywhere.** Block bootstrap (8-week blocks) for lift intervals; paired bootstrap for differences between variants; Wilson intervals for calibration bins.
- **Be explicit about forking paths.** Several variants were tried, so any "winner" is a hypothesis, not a result (see below).

## Results (backtest 2020–2024, 261 weeks)

All intervals are 90% block-bootstrap intervals. Samples are small (47 spike weeks at the 18% definition), so every number is noisy.

| Question | Result |
|---|---|
| Does LightGBM beat persistence on average error (H=4)? | **No.** MAE ratio vs persistence averages 1.19 to 1.24, worse in all 5 test years. |
| Spike ranking, H=4 (lift) | LightGBM **1.4** (1.1 to 2.2); simple 4-week-change rule **1.5** (1.1 to 2.2). No clear difference. |
| Spike ranking, H=8 (lift) | LightGBM **1.8** (1.3 to 2.8); simple rule **0.9** (0.8 to 1.1). |
| Are the default classifier's probabilities trustworthy? (H=4, spike = rise above 25%, base rate 18%) | **No, they are overconfident.** Weeks where it said about 2% (bin up to 10%): 16% actually spiked (33 of 202 weeks). Weeks where it said about 82% (bin above 40%): 30% spiked (3 of 10; 90% interval 13% to 56%). |

**Reading this honestly.** There is a little ranking information (lift about 1.4 to 1.8, intervals mostly above 1), but not enough to act on, and it is not clearly better than a one-line rule at H=4. Regression predictions also "shrink" toward the average, so they work as scores for ranking, not as price forecasts.

### A hypothesis, not a finding

Removing the two holiday-calendar features (weeks relative to Idul Fitri and Idul Adha) raised lift: H=4 from 1.36 to 1.87, H=8 from 1.78 to 2.81, with paired-bootstrap differences of +0.47 (+0.11 to +0.95) and +0.98 (+0.15 to +2.32). But this was found after trying several variants on the same data, which inflates apparent winners (winner's curse). I treat it as a hypothesis and test it forward: both versions are logged side by side (see below).

## Prospective predictions (the part that cannot be faked)

A GitHub Actions workflow (`.github/workflows/predict.yml`) runs whenever a new PIHPS file is added to `data/raw/` (or manually). It trains on data available up to that week, appends predictions to `predictions/log.csv`, and commits the file. The commit history proves each row existed **before** its target week happened.

Systems logged side by side for every reference week and horizon:

| System | Meaning |
|---|---|
| `lgbm_semua` | LightGBM, all features |
| `lgbm_tanpa_kalender` | LightGBM without the holiday-calendar features |
| `aturan_ubah4` | Simple rule: price change over the last 4 weeks |
| `baseline_persisten` | Persistence (score 0) |

Log columns: `tanggal_prediksi`, `minggu_acuan`, `H`, `minggu_target`, `sistem`, `skor`, `harga_acuan`. Writing is idempotent: re-running never duplicates a row. The first rows are dated 2026-10-23 (reference week ending 2026-10-18; targets 2026-11-15 and 2026-12-13). Scoring code and results will be added once enough target weeks have passed; an honest verdict needs months, not weeks.

The same predictions were produced on two different machines (my sandbox and the GitHub runner) and matched to four decimals, which is a basic reproducibility check.

## Repository layout

```
src/bersihkan.py   read a PIHPS xlsx and return one commodity's daily price series
src/fitur.py       weekly aggregation and features (lags, changes, 52-week mean, holiday weeks)
src/prediksi.py    train, score the latest complete week, append to the log
data/raw/          PIHPS exports (full history, filenames end with the last date)
predictions/       append-only prediction log
requirements.txt   pinned versions
```

Run from the repository root:

```
pip install -r requirements.txt
python src/prediksi.py
```

## Limitations

- **Small samples.** 47 spike weeks in five test years; intervals are wide.
- **One market, one commodity.** Kota Depok, curly red chili only.
- **Backtest is not pristine.** The 2025+ period was opened once for early checks, so the forward log is the clean test.
- **Multiple comparisons.** Several variants were tried on the same backtest.
- **Holiday dates.** Some Idul Fitri and Idul Adha dates in `src/fitur.py` are not yet verified against an official source.
- **Weekday-only data**, so a week counts as complete once data reaches Friday.
- **Data rights.** PIHPS has no documented terms or API that I could find. The raw export is stored in this repository only so the workflow can read it, and the repository is currently private.
- **Experiments not yet included.** The exploratory notebooks are not in this repo yet.

## Ringkasan (Bahasa Indonesia)

Proyek ini mencoba membuat sinyal peringatan dini lonjakan harga cabai merah keriting di Depok (4 dan 8 minggu ke depan) dengan data PIHPS. Hasil uji mundur 2020–2024 menunjukkan **sinyal lemah**: model hanya mengurutkan minggu lonjakan sekitar 1,4 kali lebih baik dari acak, dan tidak mengalahkan tebakan "harga tetap" pada rata-rata kesalahan. Kekuatan proyek ini ada pada evaluasi yang jujur (pemisahan data bersih, interval ketidakpastian) dan **log prediksi ke depan** yang dicatat otomatis oleh robot GitHub sebelum kenyataannya diketahui. Dugaan bahwa fitur kalender hari raya merugikan baru berupa hipotesis dan sedang diuji lewat log tersebut.
