# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:03:27 2021

@author: joshu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import re
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('./boston/reviews.csv')
reviews.head()
reviews.columns


df_reviews = df[['review_scores_rating',  'review_scores_value', 'reviews_per_month']]

df = df.fillna(value={"neighborhood_overview": ''})
df['price_'] = df['price'].replace('[\$,]', '', regex=True).astype(float)


#Can you describe the vibe of each Boston neighborhood using listing descriptions?
stopwords_eng = stopwords.words('english')
stopwords_in_context = ['many','several','local','short','great','whole','next']
stopwords_eng.extend(stopwords_in_context)

neighbourhoods = df['neighbourhood_cleansed'].unique().tolist()
neighbourhoods_keywords = {}
for n in neighbourhoods:
    neighborhood_overview = ' '.join(df[df['neighbourhood_cleansed']==n]["neighborhood_overview"])
    pattern = re.compile(r'\b(' + r'|'.join(stopwords_eng) + r')\b\s*')
    neighborhood_overview = pattern.sub('', neighborhood_overview.lower())
    words = re.findall(r'\w+', neighborhood_overview)
    words_common = collections.Counter(words).most_common(100)
    top_words = []
    for w in words_common:
        text = nltk.word_tokenize(w[0])
        tag = nltk.pos_tag(text)
        if tag[0][1] == 'JJ':
            top_words.append(w)  
        if len(top_words) >= 6:
            break
    neighbourhoods_keywords[n] = top_words

print(neighbourhoods_keywords)



# Find the ten most common words in Hamlet

neighborhood_overview = ' '.join(df["neighborhood_overview"])
pattern = re.compile(r'\b(' + r'|'.join(stopwords_eng) + r')\b\s*')
neighborhood_overview = pattern.sub('', neighborhood_overview.lower())
words = re.findall(r'\w+', neighborhood_overview)
words_common = collections.Counter(words).most_common(30)
top_words = []
for w in words_common:
    top_words.append(w[0])


def contiansKeywords(string, keywords):
    for word in keywords:
        if word in string:
            return True
    return False

def contiansKeywordsCount(string, keywords):
    count = 0
    for word in keywords:
        if word in string:
            count = count + 1
    return count        

df['keywords_related'] = df['neighborhood_overview'].apply(contiansKeywords, args=(top_words,))
df['keywords_count'] = df['neighborhood_overview'].apply(contiansKeywordsCount, args=(top_words,))
group_data2 = df.groupby(['keywords_related','neighbourhood_cleansed'],as_index=False).agg({'price_':'mean'})
group_data3 = df.groupby(['keywords_count','neighbourhood_cleansed'],as_index=False).agg({'price_':'mean'})



#neighborhood_overview
#transit
#access
#house_rules  ['smoking','pet','parties','event','shoes','drug','vegetarian','quiet','alcohol']  
#host_location
#host_response_time
#host_response_rate
#host_acceptance_rate
#host_neighbourhood  
#street
#neighbourhood
#cancellation_policy
#review_scores_rating
#review_scores_cleanliness

#price
#cleaning_fee
#room_type
#property_type
#market = 'Boston'


