import pandas as pd
import numpy as np
import datetime as dt
import calendar

# This is a list of start dates determined manually based on the cleanliness of data after the official commision date.
cleanDictDefault = {'BULGANA1':dt.datetime(2021,10,1), 
                    'WINTSF1':dt.datetime(2021,10,1), 
                    'MRTLSWF1':dt.datetime(2023,7,1), 
                    'COHUNSF1':dt.datetime(2023,1,1), 
                    'DUNDWF3':dt.datetime(2021,10,1),
                    'BRYB1WF1':dt.datetime(2021,10,1), 
                    'GLENSF1':dt.datetime(2024,4,1),
                    'Fulham Solar Farm':dt.datetime(2027,11,17)}

def filter_generation_data(generationData:pd.DataFrame, exclusionPeriods:pd.DataFrame, assets:list,
                           extendExclusions:bool = False, cleanDict:dict = cleanDictDefault):
    startDate = min(cleanDict.values())
    generationData = generationData[generationData.index>=startDate]

    # Extend exclusion periods to entire month when more than 10 days is excluding
    if extendExclusions:
        print("Extending exclusions")
        exclusionPeriods = extend_exclusion_periods(exclusionPeriods)

    filteredData = pd.DataFrame(index=generationData.index)
    for asset in assets:
        df = generationData[asset]

        # Define the start date as the first non-NaN value
        startDate = df.dropna().index.min()
        df = df.loc[startDate:]
        print("{} : {} to {} : {} 30-min intervals".format(asset, df.index.min().strftime("%d-%m-%Y"), df.index.max().strftime("%d-%m-%Y"), len(df)))

        # Filter initial commisioning period
        filterDate = (df.index >= cleanDict[asset])        

        # Exclude periods of obvious operational adjustments
        if asset in list(exclusionPeriods['DUID'].unique()):
            for idx, exclusion in exclusionPeriods[exclusionPeriods['DUID'] == asset].iterrows():
                filterExclusion = ((df.index >= exclusion['ExclusionStart']) & (df.index <= exclusion['ExclusionEnd']))
                filterDate = filterDate & ~filterExclusion
            print("...data points filtered for commissioning & exclusions : {} ({}%)".format(len(df) - filterDate.sum(),np.round(((len(df) - filterDate.sum())/len(df))*100)))
        
        dfAsset = df[filterDate] 

        filteredData = pd.concat([filteredData, dfAsset], axis=1)

    return [filteredData, exclusionPeriods]


def extend_exclusion_periods(exclusionPeriods:pd.DataFrame):

    extended_exclusionPeriods = exclusionPeriods.copy()

    for i in exclusionPeriods.index:
        exRange = pd.date_range(exclusionPeriods.loc[i, 'ExclusionStart'], exclusionPeriods.loc[i, 'ExclusionEnd'])
        days_per_month = exRange.to_series().groupby([exRange.year, exRange.month]).count()

        exclude_months = days_per_month[days_per_month >= 10]

        # Keep all exclusion periods which don't have > 10days periods each month
        if exclude_months.empty:
            extended_exclusionPeriods.loc[i] = exclusionPeriods.loc[i]
            continue

        # For each exclusion period which has >10 days either at the start of the end
        # Check if start date should be extended
        if days_per_month.iloc[0] >= 10:
            extended_exclusionPeriods.loc[i,'ExclusionStart'] = exclusionPeriods.loc[i,'ExclusionStart'].replace(day=1)

        # Check if end date should be extended
        if days_per_month.iloc[-1] >= 10:
            year = exclusionPeriods.loc[i,'ExclusionEnd'].year
            month = exclusionPeriods.loc[i,'ExclusionEnd'].month
            day = calendar.monthrange(year,month)[1]
            extended_exclusionPeriods.loc[i,'ExclusionEnd'] = extended_exclusionPeriods.loc[i,'ExclusionEnd'].replace(day=day)

        print(f"...exclusion period {i} updated to {extended_exclusionPeriods.loc[i,'DUID']} {extended_exclusionPeriods.loc[i,'ExclusionStart']} to {extended_exclusionPeriods.loc[i,'ExclusionEnd']}")

    return extended_exclusionPeriods
