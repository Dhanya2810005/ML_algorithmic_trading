import numpy as np
import pandas as pd

def compute_rsi(series, period=14):
    delta = series.diff().fillna(0)
    gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def compute_stochastic_oscillator(df, period=14, smooth_k=3, smooth_d=3):
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    df['%K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['%D'] = df['%K'].rolling(window=smooth_d).mean()
    return df

def compute_adx(df, period=14):
    df['+DM'] = df['High'].diff()
    df['-DM'] = df['Low'].diff() * -1
    df['+DM'] = np.where((df['+DM'] > df['-DM']) & (df['+DM'] > 0), df['+DM'], 0)
    df['-DM'] = np.where((df['-DM'] > df['+DM']) & (df['-DM'] > 0), df['-DM'], 0)
    df['TR'] = np.maximum(df['High'] - df['Low'],
                          np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                     abs(df['Low'] - df['Close'].shift(1))))
    df['+DM_EMA'] = df['+DM'].ewm(span=period, adjust=False).mean()
    df['-DM_EMA'] = df['-DM'].ewm(span=period, adjust=False).mean()
    df['TR_EMA'] = df['TR'].ewm(span=period, adjust=False).mean()
    df['+DI'] = (df['+DM_EMA'] / df['TR_EMA']) * 100
    df['-DI'] = (df['-DM_EMA'] / df['TR_EMA']) * 100
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    df['ADX'] = df['DX'].ewm(span=period, adjust=False).mean()
    return df

def create_features(df):
    df = df.copy()
    df['MA50'] = df['Close'].rolling(50, min_periods=1).mean()
    df['MA200'] = df['Close'].rolling(200, min_periods=1).mean()
    df['MA_Crossover'] = (df['MA50'] > df['MA200']).astype(int)
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()
    df['SD20'] = df['Close'].rolling(20, min_periods=1).std()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df = compute_stochastic_oscillator(df)
    df = compute_adx(df)
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    return df
