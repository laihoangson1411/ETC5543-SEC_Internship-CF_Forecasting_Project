import pandas as pd

# ---------------------------------------------------------------------------
# Date helpers: financial year (FY) and season
# ---------------------------------------------------------------------------

def getFY(ds):  
    """ Derive Financial Year from Datetime"""
    return pd.PeriodIndex(ds, freq='Y-JUN').strftime('%Y').astype(int)



def getSeason(ds):
    """ Derive Season from Datetime (1=Summer, 2=Autumn, 3=Winter, 4=Spring)"""
    s = ds.month.astype(int)
    season = s%12//3+1
    return season


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------
def convertUnits(df:pd.DataFrame, origUnits:str, outputUnits:str):
    '''
    Can be called to convert the units of a generation/load dataframe
    INPUTS:
        df:pd.DataFrame - a load or generation forecast/historic dataset
        origUnits:str   - original units
        outputUnits:str - desired output units
    OUTPUTS:
        df:pd.DataFrame - dataframe with units of outputUnits
    '''
    # Assert statement to ensure allowable units are defined
    assert origUnits in ['kWh', 'kW', 'MWh', 'MW', 'TW']
    assert outputUnits in ['kWh', 'kW', 'MWh', 'MW', 'TW']

    # Dictionary to establish scaling
    unitDict = {'kWh': 1e3,
                'kW': 1e3,
                'MWh': 1e6,
                'MW': 1e6,
                'TW': 1e9}

    # Determine the interval of the incoming dataset and check if it is uniform
    intervals = df.index.to_series().diff().dropna()
    if intervals.nunique() != 1:
        raise Exception('Datetime index is not uniform - check for missing data!')

    energy, power = ['kWh', 'MWh'], ['kW', 'MW']
    if (origUnits in energy and outputUnits in energy) or (origUnits in power and outputUnits in power):
        # Energy to energy or power to power: simple conversion factor
        conversionFactor = unitDict[origUnits] / unitDict[outputUnits]
    elif origUnits in ['kWh', 'MWh'] and outputUnits in ['kW', 'MW', 'TW']:
        # If converting Energy to Power, apply interval conversion
        conversionFactor = pd.Timedelta(hours=1) / intervals.iloc[0] * (unitDict[origUnits] / unitDict[outputUnits])
    elif origUnits in ['kW', 'MW', 'TW'] and outputUnits in ['kWh', 'MWh']:
        # If converting Power to Energy, apply interval conversion
        conversionFactor = intervals.iloc[0] / pd.Timedelta(hours=1) * (unitDict[origUnits] / unitDict[outputUnits])
    
    df = df * conversionFactor

    return df


# Default aggregation methods for contract factor columns.
cols_default = {'': 'Sum',
                '_TWAP': 'WeightedMean',
                '_ContractRevenue': 'Sum',
                '_MarketRevenue': 'Sum',
                '_CapFactor': 'WeightedMean',
                '_TotalPeriods': 'Sum',
                '_ContractDWAP': 'WeightedMean',
                '_DWAP': 'WeightedMean',
                '_ContractFactor': 'WeightedMean',
                '_GenFactor': 'WeightedMean'}

def expfunctionCF(x, b, m):
    """
    Calculate the output of a simple exponential function used for forecasting contract factors.

    Parameters:
        x (float): Exponent value.
        b (float): Base multiplier.
        m (float): Exponential base.

    Returns:
        float: Result of b * m**x.
    """
    y = b * m**x
    return y


def custom_agg(group, asset=None):
    '''
    Apply custom aggregation to columns of a pandas DataFrame for contract factor calculations.

    Parameters:
        group (pd.DataFrame): DataFrame group to aggregate.
        asset (str, optional): Asset column name for counting periods.

    Returns:
        pd.Series: Aggregated values for each column.
    '''
    d = {}
    for col in group.columns:
        if col in ['VIC1Price'] or col.endswith(('CapFactor', 'DWAP', 'TWAP', 'TempMean')):
            # Calculate average Price, Capacity Factor, DWAP and TWAP
            d[col] = group[col].mean()
        elif col.endswith('LogReturns'):
            # Calculate log-return volatility
            d[col] = group[col].std()
        else:
            # Otherwise calculate aggregated total
            d[col] = group[col].sum()
    if asset:
        d[f'{asset}_TotalPeriods'] = group[asset].count()
    
    return pd.Series(d)


def calcMonthlyContractFactors(generationTrace, assets, capacity, extraCol=None, zeroDollarFloor:bool = True):
    '''
    Calculate generation and contract factors at a monthly level from half-hourly generation trace.

    Parameters:
        generationTrace (pd.DataFrame): Half-hourly generation trace data.
        assets (list): List of asset column names.
        capacity (dict): Dictionary mapping asset names to their capacity (MW).
        extraCol (str, optional): Additional column to include in calculations.
        zeroDollarFloor (bool): If True, apply zero dollar floor to contract revenue.

    Returns:
        pd.DataFrame: Monthly aggregated contract factors for each asset.
    '''
    generationTraceResampledAgg = pd.DataFrame()
    generationTraceResampledAgg['VIC1Price_TWAP'] = generationTrace.loc[:,['VIC1Price']].resample('ME').mean()
    
    # Add an additional column into the resampled data (Only one additional column able to be provided for now)
    for asset in assets:
        if extraCol:
            tempGenerationTrace = generationTrace[[asset, 'VIC1Price', f'{asset}{extraCol}']]
        else:
            tempGenerationTrace = generationTrace[[asset, 'VIC1Price']]
        
        # Calculate Revenue (Volume * Price); the contract revenue floors the price at $0 if zeroDollarFloor
        contract_price = tempGenerationTrace['VIC1Price'].clip(lower=0) if zeroDollarFloor else tempGenerationTrace['VIC1Price']
        tempGenerationTrace.loc[:,[f'{asset}_ContractRevenue']] = tempGenerationTrace[asset] * contract_price
        tempGenerationTrace.loc[:,[f'{asset}_MarketRevenue']] = tempGenerationTrace[asset] * tempGenerationTrace['VIC1Price']
        
        # Calculate Capacity Factor (Convert Volume to MW / Capacity in MW)
        tempGenerationTrace.loc[:,[f'{asset}_CapFactor']] = convertUnits(tempGenerationTrace[asset], origUnits='MWh', outputUnits='MW') / capacity[asset]

        # Remove the price from Asset_TWAP for any periods that the asset was offline
        tempGenerationTrace.loc[:,['VIC1Price']] = tempGenerationTrace['VIC1Price'].where(~tempGenerationTrace[asset].isna())

        generationTraceResampled = tempGenerationTrace.resample('ME').apply(custom_agg, asset)
        generationTraceResampled = generationTraceResampled.rename(columns={'VIC1Price': f'{asset}_TWAP'})
        
        # With the resampled data calculate DWAP (Revenue / Volume)
        generationTraceResampled.loc[:,f'{asset}_ContractDWAP'] = generationTraceResampled[f'{asset}_ContractRevenue'] / generationTraceResampled[asset]
        generationTraceResampled.loc[:,f'{asset}_DWAP'] = generationTraceResampled[f'{asset}_MarketRevenue'] / generationTraceResampled[asset]
        # With the resampled data calculate GenFactor (DWAP / TWAP)
        generationTraceResampled.loc[:,f'{asset}_ContractFactor'] = generationTraceResampled[f'{asset}_ContractDWAP'] / generationTraceResampled[f'{asset}_TWAP']
        generationTraceResampled.loc[:,f'{asset}_GenFactor'] = generationTraceResampled[f'{asset}_DWAP'] / generationTraceResampled[f'{asset}_TWAP']

        generationTraceResampledAgg = pd.concat([generationTraceResampledAgg, generationTraceResampled], axis=1)

    return generationTraceResampledAgg


def aggregateContractFactorsByGenType(weightedContractFactors, assets, genType, cols:dict = cols_default):
    '''
    Aggregate weighted contract factors to generation type level.

    Parameters:
        weightedContractFactors (pd.DataFrame): DataFrame of weighted contract factors.
        assets (list): List of asset column names.
        genType (str): Generation type name.
        cols (dict): Dictionary specifying aggregation method for each column.

    Returns:
        pd.DataFrame: DataFrame with aggregated contract factors by generation type.
    '''
    # Use the cols dictionary to calculate weighted average or sum
    for col in cols.keys():
        assetsKey = [asset + col for asset in assets]
        if cols[col] == 'Sum':
            weightedContractFactors[f'{genType}{col}'] = weightedContractFactors.loc[:,assetsKey].sum(axis=1)
        elif cols[col] == 'WeightedMean':
            # Calculate dot product using mul method then sum across cols before dividing by total generation
            weightedContractFactors[f'{genType}{col}'] = weightedContractFactors[assetsKey].mul(weightedContractFactors[assets].values).sum(axis=1) / weightedContractFactors[assets].sum(axis=1)

    return weightedContractFactors
