# data/tourism/

Tourism Monthly dataset (Monash Time Series Forecasting Archive).

- **Source:** https://zenodo.org/record/4656096 (`tourism_monthly_dataset.zip`)
- **Origin:** Kaggle tourism forecasting competition (Athanasopoulos et al.,
  IJF 2011); 366 monthly series, 91–333 obs each, standard horizon h=24.
- **Used by:** exp60 (full ladder + proposed method): proposed 0.1282 beats
  equal weights 0.1869 (p=1.9e-68) and best single snaive 0.1387 (p=2.6e-34).
  7-model run: timesfm25 0.0965 CRPS (uncontaminated — tourism in neither
  TSFM training corpus); proposed 0.0974 ties best single (p=0.42).
- **Note:** data files are gitignored; this README is force-added. Re-download
  from Zenodo to reproduce.
