# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 01:34:06 2021

@author: joshu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections


df = pd.read_csv('./calendar.csv')
df.head()


#What are the busiest times of the year to visit Boston? By how much do prices spike?
#listing_id        date available price

grouped = df.groupby(['date','available']).count()

.groupby(['col1','col2']).mean()

#g = grouped.get_group(12147973)


print(df.count())