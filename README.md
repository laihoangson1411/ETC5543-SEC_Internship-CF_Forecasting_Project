# ETC5543 - SEC Internship - CF Forecasting Project
```
ETC5543-SEC_Internship-CF_Forecasting_Project/
├── README.md
├── data/
│   ├── raw/
│   │   ├── HistoricLGCPrice.csv # 20260424_HistoricLGCPrice.csv
│   │   ├── ForecastedLGCPrice.csv # new file
│   │   ├── MLF_FY23-26.csv # AEMO_VICMLF_FY23-26.csv
│   │   ├── HistoricWeather_30min.csv # dtn_hist_30min.csv
│   │   ├── ExclusionPeriods.csv
│   │   ├── GenerationMW_FY19-26.csv # NEMSight_VICVREGeneration_FY19-26_MW.csv
│   │   ├── HistoricSolarIrradiance_30min.csv # solcast_hist_30min.csv
│   │   ├── PriceForecast_30min.csv # VIC_Central_30min_PriceForecast.csv
│   │   └── Asset.csv # VRETAsset.csv
│   ├── processed/
│   │   ├── QM_HistoricWeather_30min.csv # dtn_weather_89002_86338_qm.csv, write csv from Wind_QuantileMapping.qmd
│   │   └── SolarProxy.csv # Write csv from SolarProxy_Cal.qmd 
│   └── forecast/ #both WIND and SOLAR in one csv. Example: after get the result, append it to the csv
│   │   ├── TestFY26_Linear.csv # Write csv from _Analysis.qmd
│   │   ├── TestFY26_XGB.csv # Write csv from _Analysis.qmd
│   │   ├── TestFY26_Ridge.csv # Write csv from _Analysis.qmd
│   │   ├── TestFY26_SNaive.csv # Write csv from _Analysis.qmd
│   │   ├── TestFY26_Exponential.csv # Write csv from _Analysis.qmd
│   │   ├── TestFY26_RF.csv # Write csv from _Analysis.qmd
│   │   ├── TestFY26_Ensemble.csv # Write csv from _Analysis.qmd
│   │   ├── FY27-29_Ridge.csv # Write csv from CF_Forecast_FY27-29.qmd
│   │   └── FY27-29_XGB.csv # Write csv from CF_Forecast_FY27-29.qmd
├── functions/
│   ├── utilsContractFactors.py
│   ├── utilsData.py
│   ├── utilsGeneration.py
│   ├── utilsModels.py
│   └── utilsScenario.py
├── analysis/
│   ├── IDA_ContractFactors.qmd
│   ├── IDA_Weather_SolarIrradiance.qmd
│   ├── EDA.qmd # Also include LGC's EDA in here
│   ├── Wind_QuantileMapping.qmd
│   ├── SolarProxy_Cal.qmd 
│   ├── WindCF_Analysis.qmd
│   ├── SolarCF_Analysis.qmd
│   └── CF_Forecast_FY27-29.qmd
├── images/
│   ├── AnalysisFramework.png
│   ├── ForecastFramework.png
│   └── ...
├── report.qmd
├── report.pdf
├── presentation.qmd
├── presentation.pdf
├── css/
│   ├── report.css
│   └── presentation.css
├── LICENSE.md
└── .gitignore
```