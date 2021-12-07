# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:03:27 2021

@author: joshu
"""

import sys
print(sys.executable)

#/anaconda3/bin/python -m pip install wordcloud

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
stop_words = set(stopwords.words('english'))

df = pd.read_csv('./boston/listings.csv')
df.head()
df.columns


df_reviews = df[['review_scores_rating',  'review_scores_value', 'reviews_per_month']]

df = df.fillna(value={"neighborhood_overview": ''})
df['price_'] = df['price'].replace('[\$,]', '', regex=True).astype(float)


#Which neighborhood has the highest number of listings in Boston?
feq = df['neighbourhood'].value_counts().sort_values(ascending=True)
feq.plot.barh(figsize=(10,8), color ='b', width= 1)
plt.title('Number of listings by neighbourhood', fontsize=14)
plt.xlabel('Number of listings', fontsize = 12)
plt.show()


#Can you describe the vibe of each Boston neighborhood?
#Here I will be picking top 10 Boston neighborhood with most number of listings
stopwords_eng = stopwords.words('english')
stopwords_in_context = ['many','several','local','short','great','whole','next']
stopwords_eng.extend(stopwords_in_context)

feq = df['neighbourhood'].value_counts().sort_values(ascending=False)[:10]



#neighbourhoods = df['neighbourhood'].unique().tolist()
neighbourhoods = feq.keys().values.tolist()
neighbourhoods_keywords = {}
for n in neighbourhoods:
    neighborhood_overview = ' '.join(df[df['neighbourhood']==n]["neighborhood_overview"])
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
        if len(top_words) >= 10:
            break
    neighbourhoods_keywords[n] = top_words

print(neighbourhoods_keywords)

#convert neighbourhoods_keywords into dict of dict
neighbourhoods_keywords_dict = {}
for k,v in neighbourhoods_keywords.items():
    print(k)
    inner_dict = {}
    for item in v:
        print(item[0])
        inner_dict[item[0]] = item[1]
    neighbourhoods_keywords_dict[k] = inner_dict 


for k,v in neighbourhoods_keywords_dict.items():
   wordcloud = WordCloud(width=800, height=400)
   wordcloud.generate_from_frequencies(frequencies=v)
   plt.figure( figsize=(6,2) )
   plt.imshow(wordcloud, interpolation="bilinear")
   plt.title('Keywords for ' + k, fontsize=14)
   plt.axis("off")
   plt.show() 
      
#Create the word cloud from the file we have 
cvec_dict = dict(zip(cvec_df.words, cvec_df.counts))

wordcloud = WordCloud(width=800, height=400)
wordcloud.generate_from_frequencies(frequencies=cvec_dict)
plt.figure( figsize=(20,10) )
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


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


