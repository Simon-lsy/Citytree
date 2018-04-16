import matplotlib.pylab as plt
import numpy as np
import random
import pandas as pd
from pandas import DataFrame

df = pd.read_csv('period_series.csv')


type_list = [3, 4]

for type in type_list:
    new_df = df[df.type == type].copy()
    print(len(new_df))

    period_series = new_df['period_series'].values

    series_312_310 = 0
    series_310_735 = 0
    series_735_513 = 0
    for series in period_series:

        if '312-310' in series:
            series_312_310 += 1
        if '310-735' in series:
            series_310_735 += 1
        if '735-513' in series:
            series_735_513 += 1

    print(series_312_310/len(new_df))
    print(series_310_735/len(new_df))
    print(series_735_513/len(new_df))
    print('----------------')

