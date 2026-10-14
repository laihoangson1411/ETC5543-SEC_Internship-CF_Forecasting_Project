# ETC5543 - SEC Internship - CF Forecasting Project
ETC5543-SEC_Internship-CF_Forecasting_Project/
├── README.md
├── data/
│   ├── raw/
│   │   ├── HistoricLGCPrice.csv
│   │   ├── ForecastedLGCPrice.csv
│   │   ├── MLF_FY23-26.csv
│   │   ├── HistoricWeather_30min.csv
│   │   ├── ExclusionPeriods.csv
│   │   ├── GenerationMW_FY19-26.csv
│   │   ├── HistoricSolarIrradiance_30min.csv
│   │   ├── PriceForecast_30min.csv
│   │   └── Asset.csv
│   ├── processed/
│   │   ├── QM_HistoricWeather_30min.csv
│   │   └── SolarProxy.csv
│   └── forecast/ #both WIND and SOLAR in one csv. Example: after get the result, append it to the csv
│   │   ├── TestFY26_Linear.csv 
│   │   ├── TestFY26_XGB.csv
│   │   ├── TestFY26_Ridge.csv
│   │   ├── TestFY26_SNaive.csv
│   │   ├── TestFY26_Exponential.csv
│   │   ├── TestFY26_RF.csv
│   │   ├── TestFY26_Ensemble.csv
│   │   ├── FY27-29_Ridge.csv
│   │   └── FY27-29_XGB.csv
├── functions/
│   ├── utilsContractFactors.py
│   ├── utilsData.py
│   ├── utilsGeneration.py
│   ├── utilsModels.py
│   └── utilsScenario.py
├── notebooks/
│   ├── IDA_ContractFactors.qmd
│   ├── IDA_Weather_SolarIrradiance.qmd
│   ├── EDA.qmd
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
└── .gitignore
