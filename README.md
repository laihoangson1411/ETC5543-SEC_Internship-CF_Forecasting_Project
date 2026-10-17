# ETC5543 - SEC Internship - CF Forecasting Project

Forecast of the monthly **contract factor** (CF = DWAP / TWAP, negative prices floored at \$0) of SEC's VRET wind and solar portfolios for FY27-29, using weather, the price structure and the LGC certificate price.

## Repository structure

```
ETC5543-SEC_Internship-CF_Forecasting_Project/
├── _quarto.yml                  # Quarto project: render order of the whole pipeline
├── requirements.txt             # pinned Python packages
├── README.md
├── data/
│   ├── raw/                     # inputs (original file name in brackets)
│   │   ├── Asset.csv                          (VRETAsset.csv)
│   │   ├── ExclusionPeriods.csv
│   │   ├── ForecastedLGCPrice.csv
│   │   ├── GenerationMW_FY19-26.csv           (NEMSight_VICVREGeneration_FY19-26_MW.csv), time-starting
│   │   ├── HistoricLGCPrice.csv               (20260424_HistoricLGCPrice.csv)
│   │   ├── HistoricSolarIrradiance_30min.csv  (solcast_hist_30min.csv)
│   │   ├── HistoricWeather_30min.csv          (dtn_hist_30min.csv)
│   │   ├── Locations.csv                      coordinates of the weather stations and farms
│   │   ├── MLF_FY23-26.csv                    (AEMO_VICMLF_FY23-26.csv)
│   │   ├── PriceForecast_30min.csv            (VIC_Central_30min_PriceForecast.csv), time-ending
│   │   └── PriceForecast_Month.csv            (20261001_ASXadjustedPrice.csv)
│   ├── processed/               # written by the analysis notebooks
│   │   ├── HistoricContractFactors.csv        <- IDA_ContractFactors.qmd
│   │   └── QM_HistoricWeather_30min.csv       <- Wind_QuantileMapping.qmd
│   └── forecast/                # FY27-29_CF_Forecast.csv <- CF_Forecast_FY27-29.qmd (chosen model of wind and solar)
├── functions/                   # Python modules shared by the notebooks
│   ├── utilsConfig.py           # paths and the shared configuration (cfg): window, assets, stations, features, models
│   ├── utilsData.py             # loading generation and weather, monthly feature helpers
│   ├── utilsContractFactors.py  # contract factor calculation
│   ├── utilsGeneration.py       # commissioning and exclusion periods
│   ├── utilsModels.py           # models, Exponential baseline, metrics
│   ├── utilsScenario.py         # price forecast, LGC price and shrink factor, scenario rows
│   └── utilsPlot.py             # colours, plot style, titles, shared plots
├── analysis/                    # analysis notebooks (render options in analysis/_metadata.yml)
│   ├── IDA_ContractFactors.qmd
│   ├── IDA_Weather_SolarIrradiance.qmd
│   ├── Wind_QuantileMapping.qmd
│   ├── EDA.qmd
│   ├── WindCF_Analysis.qmd
│   ├── SolarCF_Analysis.qmd
│   └── CF_Forecast_FY27-29.qmd
├── images/                      # pictures used by the presentation
├── css/presentation.css         # slide styles
├── report.qmd
└── presentation.qmd
```

## Set-up

1. **Python 3.13** with the pinned packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
   On macOS XGBoost also needs OpenMP: `brew install libomp`.
2. **Quarto 1.8** or later (<https://quarto.org>). For the PDF version of the report: `quarto install tinytex`.

## Reproducing the results

Every notebook reads the files written by the ones before it, so they must run in this order. From the project folder:

```bash
quarto render                                   # the whole pipeline, in the order of _quarto.yml
quarto render analysis/WindCF_Analysis.qmd      # or a single file
```

| Order | File | Purpose | Writes |
|---|---|---|---|
| 1 | `analysis/IDA_ContractFactors.qmd` | Monthly CF of every asset and portfolio, data checks | `data/processed/HistoricContractFactors.csv` |
| 2 | `analysis/IDA_Weather_SolarIrradiance.qmd` | Choice of weather stations and variables, data quality | - |
| 3 | `analysis/Wind_QuantileMapping.qmd` | Correction of the corrupted DTN wind speed (Oct 2025 - Apr 2026) | `data/processed/QM_HistoricWeather_30min.csv` |
| 4 | `analysis/EDA.qmd` | CF drivers, Exponential baseline, LGC and negative prices | - |
| 5 | `analysis/WindCF_Analysis.qmd` | Wind models, FY26 test with the scenario forecast | - |
| 6 | `analysis/SolarCF_Analysis.qmd` | Solar models, FY26 test with the scenario forecast | - |
| 7 | `analysis/CF_Forecast_FY27-29.qmd` | Retrain on FY23-26 and forecast FY27-29 (Ensemble for wind, RF for solar) | `data/forecast/FY27-29_CF_Forecast.csv` |
| 8 | `report.qmd`, `presentation.qmd` | Report and slides | - |

The settings shared by the notebooks (analysis window, train/test years, assets, weather stations, features and model parameters) are in `functions/utilsConfig.py`.

## Notes on reproducibility

- **Use the pinned versions.** The model results, XGBoost in particular, change between library versions even with the fixed seeds (`random_state = 42`).
- **Render with Quarto.** `quarto render` runs each notebook from the top on a fresh kernel. Results read from cells run one by one in an editor can mix old and new code after an edit.
- **Do not use `--output-dir`** with `quarto render`: in a project it also removes the HTML files rendered next to the sources.
- **Timestamps.** The generation file is time-starting; the 30-min price forecast is time-ending and is moved back 30 minutes when it is read (`utilsScenario.load_forecast_price_hh`).
- Small differences (last decimals) between computers are expected in `QM_HistoricWeather_30min.csv` when it is rebuilt; they move the model errors only in the 4th-5th decimal.
