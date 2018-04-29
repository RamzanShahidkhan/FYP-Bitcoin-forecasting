# Import libraries
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
from datetime import datetime
warnings.filterwarnings('ignore')
plt.style.use('seaborn-poster')
import datetime as dt

# Load data
#df = pd.read_csv('../input/btceUSD_1-min_data_2012-01-01_to_2017-05-31.csv')
df = pd.read_csv('./krakenUSD_1-min_data_2014-01-07_to_2017-05-31.csv')
df = df.dropna()
df = df.astype('float64')

df_time = pd.date_range('2014-01-07','2016-05-31', freq='1m')
frame = pd.DataFrame(data=df, index=df_time)
#print(" framm ",frame)
#print(len(df))

print(df.head())

# Unix-time to
df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')

# Resampling to daily frequency
df.index = df.Timestamp
#df = df.re

df_t_m = df.resample('1T', how='mean') # minute
df_t = df.resample('60S', how='mean')
df_daily = df.resample('D').mean()
df_ohlc = df['Close'].resample('10D').ohlc()
print(len(df_ohlc))
print(df_ohlc)

print(len(df))
print(len(df_daily))
#print("d_t  ",df_t)
#print("d_mmm ", df_t_m)
print("daily  ",df_daily)


# Resampling to monthly frequency
df_month = df.resample('M').mean()
#print(len(df_month))
#print("monthly   ",df_month)
# Resampling to annual frequency
df_year = df.resample('A-DEC').mean()

# Resampling to quarterly frequency
df_Q = df.resample('Q-DEC').mean()

'''
# PLOTS
fig = plt.figure(figsize=[15, 7])
plt.suptitle('Bitcoin exchanges, mean USD', fontsize=22)

plt.subplot(221)
plt.plot(df.Weighted_Price, '-', label='By Days')
plt.legend()

plt.subplot(222)
plt.plot(df_month.Weighted_Price, '-', label='By Months')
plt.legend()

plt.subplot(223)
plt.plot(df_Q.Weighted_Price, '-', label='By Quarters')
plt.legend()

plt.subplot(224)
plt.plot(df_year.Weighted_Price, '-', label='By Years')
plt.legend()

# plt.tight_layout()
plt.show()
'''

'''
plt.figure(figsize=[15,7])
sm.tsa.seasonal_decompose()
sm.tsa.seasonal_decompose(df_month.Weighted_Price).plot()
print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.Weighted_Price)[1])
plt.show()
'''
'''
# Box-Cox Transformations
df_month['Weighted_Price_box'], lmbda = stats.boxcox(df_month.Weighted_Price)
print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.Weighted_Price)[1])
'''
