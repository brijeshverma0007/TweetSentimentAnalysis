#!/usr/bin/env python
# coding: utf-8

# # 1. Preliminary Introspection of the data

# #### Loading Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from textblob import TextBlob
import seaborn as sns
get_ipython().system('pip install varname')
from varname import nameof


# #### Creating Sentiment Analysis

# #### Loading DataFrames

# In[3]:


get_ipython().system('pip install twython')
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()


def sentiment_analysis_vader(tweet):
    score = sid.polarity_scores(tweet)["compound"]
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"


# In[4]:


get_ipython().system('pip install textblob')

from textblob import TextBlob
import tweepy


def sentiment_analysis_textblob(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"


# In[5]:


from google.colab import drive
drive.mount('/content/drive')


# In[7]:


get_ipython().run_line_magic('pwd', '')


# In[6]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive')


# In[8]:


get_ipython().system('ls')


# In[9]:


# reading the database
raw_company_db = pd.read_csv('Company.csv')
raw_company_tweet_db = pd.read_csv("Company_Tweet.csv")
raw_tweet_db = pd.read_csv("Tweet.csv")
raw_companyvalue_db = pd.read_csv("CompanyValue.csv")


# # 2. Data Cleaning

# #### Merging the DataFrames to form one single DataFrame

# In[10]:


# Part1: Merge raw_company_tweet_db, raw_company_db
raw_company_tweet_db = pd.merge(raw_company_tweet_db, raw_company_db,on='ticker_symbol')

# Part2: Merge raw_company_tweet_db, raw_company_db
raw_tweet_db = pd.merge(raw_tweet_db, raw_company_tweet_db, on="tweet_id")


# In[11]:


# Veiwing the marged Dataframe 
raw_tweet_db.head(2)


# In[12]:


# Saving the names of dataframes so we can use it if required in future
raw_tweet_db.name = nameof(raw_tweet_db)
raw_companyvalue_db.name = nameof(raw_companyvalue_db)


# In[13]:


list_df = [raw_tweet_db, raw_companyvalue_db]


# In[14]:


# Let us check the number of rows and columns in the dataframes
[print(f"For Dataframe {df.name}\nNumber of Rows are {df.shape[0]}\nNumber of Columns are {df.shape[1]}\n\n") for df in list_df]


# #### Missing values check: Let us check if there are any missing values

# In[15]:


[print(f"For DataFrame {df.name}, we have missing values check as\n{df.isna().sum()}\n\n") for df in list_df]


# In[16]:


raw_tweet_db.writer = raw_tweet_db.writer.fillna('anonymous')


# In[17]:


# let us check if all null values are replaced
raw_tweet_db.isna().sum()


# #### let us check the data types of columns

# In[18]:


[print(f"for DataFrame {df.name}\n{df.info()}\n") for df in list_df]


# # 3. Data Engineering

# ### We will convert:<br>'post_date' column in raw_tweet_db & <br>'day_date' column in raw_companyvalue_db to<br>to datetime for further processing

# In[19]:


raw_tweet_db.post_date = pd.to_datetime(raw_tweet_db.post_date, unit="s")
raw_companyvalue_db.day_date = pd.to_datetime(raw_companyvalue_db.day_date)


# In[20]:


raw_companyvalue_db.head(2)


# In[21]:


# for checking the chronologically first tweet in the dataframe, we sort the dataframe by date column
raw_tweet_db.sort_values(by="post_date", inplace=True)
raw_companyvalue_db.sort_values(by="day_date", inplace=True)


# In[22]:


raw_tweet_db.head(3)


# In[23]:


raw_tweet_db.tail(3)


# In[24]:


raw_companyvalue_db.head(3)


# In[25]:


raw_companyvalue_db.tail(3)


# ## Viweing the dates at the head and tail end of the dataframe, we see that:
# #### a. starting date to considered as 1 Jan 2019
# #### b. last tweet in raw_tweet_db dataframe was tweeted on 31 Dec 2019 
# #### c. but the stock values are given till 29 May 2020.
# 
# #### So we will drop the values for the year 2020.

# In[26]:


raw_companyvalue_db = raw_companyvalue_db[raw_companyvalue_db.day_date < "2020-01-01"]


# In[27]:


raw_companyvalue_db = raw_companyvalue_db[raw_companyvalue_db.day_date > "2019-01-01"]


# In[28]:


raw_companyvalue_db.tail()


# ### We will add additional columns to raw_companyvalue_db which we will need in future for checking the stock value performance

# In[29]:


# Let us add column to the stock price dataframe which shows the max stock price fluctuation
raw_companyvalue_db['fluctuation'] = raw_companyvalue_db.high_value - raw_companyvalue_db.low_value

# Let us add column to the stock price dataframe which shows the net rise in stock price
raw_companyvalue_db['price_gain'] = raw_companyvalue_db.close_value - raw_companyvalue_db.open_value

# Let us add column to the stock price dataframe which shows the total valuation at the end of the day
raw_companyvalue_db['total_valuation_EOD'] = raw_companyvalue_db.volume * raw_companyvalue_db.close_value


# In[30]:


raw_companyvalue_db.head(3)


# In[31]:


raw_tweet_db.count()


# In[32]:


raw_tweet_db = raw_tweet_db[(raw_tweet_db['post_date'].dt.year == 2019)]


# In[33]:


raw_tweet_db.head(3)


# ### Applying sentiment analysis to the tweets using vadern and textblob to check which would be a better fit:

# In[34]:


raw_tweet_db['sentiment_vader'] = raw_tweet_db['body'].apply(lambda x : sentiment_analysis_vader(x))


# In[35]:


raw_tweet_db['sentiment_textblob'] = raw_tweet_db['body'].apply(lambda x : sentiment_analysis_textblob(x))


# In[36]:


raw_tweet_db.head(2)


# In[37]:



mask = raw_tweet_db['sentiment_vader'] != raw_tweet_db['sentiment_textblob']

filtered_df = raw_tweet_db[mask]


# In[38]:


print(filtered_df.shape)


# In[60]:


filtered_df.to_csv('sentiment_mismatch.csv', index=False)


# ###Sentiment analysis comparision

# ####Analysing which was giving more accurate setiment analysis by comparing the tweets which had different sentiment analysis from both these 2 analysis

# ####Decided to go with vader as we found it to be more accurate in the analysis performed.

# In[39]:



sid = SentimentIntensityAnalyzer()


def sentiment_analysis(tweet):
    score = sid.polarity_scores(tweet)["compound"]
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"


# ####Dropping the sentiment_textblob column and renaming the sentiment_vader column to sentiment

# In[40]:


raw_tweet_db.drop('sentiment_textblob', axis=1, inplace=True)
raw_tweet_db.rename(columns={'sentiment_vader': 'sentiment'}, inplace=True)
raw_tweet_db.head(2)


# ### Adding a column specifying the trending score of every tweet considering the retweet count, likes and comments for that tweet

# In[41]:


# Considering there is a 'comment,  retweet & like' column, we can consider those tweets having the same sentiments
# So for counting the total number of tweets, we add a count column telling the trending score of the tweet which will be addition of all these 3 columns
# Adding 1 to trend score as tweet itself is one of the contributors to itself

raw_tweet_db.insert(7, "trend_score", raw_tweet_db.comment_num + raw_tweet_db.retweet_num + raw_tweet_db.like_num + 1)


# In[42]:


raw_tweet_db.head(2)


# ### Let us work on merging the 2 dataframes. For that, we need an anchor column to merge on

# In[43]:


print("Number of rows in dataframe:", raw_tweet_db.shape[0])


# In[45]:


# Creating Anchor Column for raw_tweet_db
raw_tweet_db.insert(3, "date_str", raw_tweet_db.post_date.astype("str").str.split(" "))
raw_tweet_db.date_str = [element[0] for element in raw_tweet_db.date_str]
raw_tweet_db.insert(0, "anchor", raw_tweet_db.date_str + raw_tweet_db.ticker_symbol)

# Creating Anchor Column for raw_companyvalue_db
raw_companyvalue_db.insert(
    2, "date_str", raw_companyvalue_db.day_date.astype("str").str.split(" ")
)
raw_companyvalue_db.date_str = [element[0] for element in raw_companyvalue_db.date_str]
raw_companyvalue_db.insert(
    0, "anchor", raw_companyvalue_db.date_str + raw_companyvalue_db.ticker_symbol
)


# In[46]:


# Merging the two dataframe
processed_db = pd.merge(raw_tweet_db, raw_companyvalue_db, on="anchor")


# In[47]:


# Let us convert he string date column "date_str_x" to datetime
processed_db.date_str_x = pd.to_datetime(processed_db.date_str_x)


# In[48]:


# Since we are aiming to see the impact of tweets on stock value (i.e. rise and fall), we can drop "neutral" sentiments
processed_db = processed_db[processed_db.sentiment != "neutral"]


# ## Making a clean database

# In[49]:


# Let us make a clean dataset with only the desired values
clean_db = processed_db[
    [
        "post_date",
        "date_str_x",
        "body",
        "trend_score",
        "ticker_symbol_x",
        "company_name",
        "sentiment",
        "close_value",
        "volume",
        "open_value",
        "high_value",
        "low_value",
        "fluctuation",
        "price_gain",
        "total_valuation_EOD"
    ]
]


# In[50]:


# Let us check how many companies do we have in our dataset

print(
    f"In our dataset, we have total {len(clean_db.company_name.value_counts())} companies, namely\n{clean_db.company_name.value_counts()}"
)


# In[51]:


# Let us check by ticker symbol
clean_db.ticker_symbol_x.value_counts()


# In[52]:


# Let us make datasets for these 5 companies
apple_df = clean_db[clean_db.ticker_symbol_x == "AAPL"]
tesla_df = clean_db[clean_db.ticker_symbol_x == "TSLA"]
amazon_df = clean_db[clean_db.ticker_symbol_x == "AMZN"]
microsoft_df = clean_db[clean_db.ticker_symbol_x == "MSFT"]


# In[53]:


# For simplicity, we will further form 2 sub dataframes per company based on the sentiments: positive and negative

pos_apple_df = apple_df[apple_df.sentiment == "positive"]
pos_tesla_df = tesla_df[tesla_df.sentiment == "positive"]
pos_amazon_df = amazon_df[amazon_df.sentiment == "positive"]
pos_microsoft_df = microsoft_df[microsoft_df.sentiment == "positive"]
neg_apple_df = apple_df[apple_df.sentiment == "negative"]
neg_tesla_df = tesla_df[tesla_df.sentiment == "negative"]
neg_amazon_df = amazon_df[amazon_df.sentiment == "negative"]
neg_microsoft_df = microsoft_df[microsoft_df.sentiment == "negative"]


# In[54]:


# Let us create dataset with limited values that give us a brief info about rise and fall in total valuation of the company over time

ovr_pos_apple_df = pos_apple_df.groupby(by=["date_str_x","fluctuation", "price_gain", "total_valuation_EOD","sentiment"], as_index=False).agg({"trend_score":pd.Series.sum})
ovr_pos_tesla_df = pos_tesla_df.groupby(by=["date_str_x","fluctuation", "price_gain", "total_valuation_EOD","sentiment"], as_index=False).agg({"trend_score":pd.Series.sum})
ovr_pos_amazon_df = pos_amazon_df.groupby(by=["date_str_x","fluctuation", "price_gain", "total_valuation_EOD","sentiment"], as_index=False).agg({"trend_score":pd.Series.sum})
ovr_pos_microsoft_df = pos_microsoft_df.groupby(by=["date_str_x","fluctuation", "price_gain", "total_valuation_EOD","sentiment"], as_index=False).agg({"trend_score":pd.Series.sum})


ovr_neg_apple_df = neg_apple_df.groupby(by=["date_str_x","fluctuation", "price_gain", "total_valuation_EOD","sentiment"], as_index=False).agg({"trend_score":pd.Series.sum})
ovr_neg_tesla_df = neg_tesla_df.groupby(by=["date_str_x","fluctuation", "price_gain", "total_valuation_EOD","sentiment"], as_index=False).agg({"trend_score":pd.Series.sum})
ovr_neg_amazon_df = neg_amazon_df.groupby(by=["date_str_x","fluctuation", "price_gain", "total_valuation_EOD","sentiment"], as_index=False).agg({"trend_score":pd.Series.sum})
ovr_neg_microsoft_df = neg_microsoft_df.groupby(by=["date_str_x","fluctuation", "price_gain", "total_valuation_EOD","sentiment"], as_index=False).agg({"trend_score":pd.Series.sum})


# # 4. Data Analysis

# ## Tesla

# ### Let us analyse for Tesla
# ### We will analyse effect of Positive Tweets on Valuation

# In[55]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.figure(figsize=(15, 6))
plt.title("Tesla: Effect of Positive Tweets on Valuation")
# since we are plotting 2 line graphs with same X-axis
ax1 = plt.gca()
ax2 = plt.twinx()


ax1.plot(
    ovr_pos_tesla_df.date_str_x,
    ovr_pos_tesla_df.total_valuation_EOD,
    color="b",
    label="Valuation",
)
ax2.plot(
    ovr_pos_tesla_df.date_str_x,
    ovr_pos_tesla_df.trend_score,
    color="g",
    label="Positive Tweets",
)

ax1.set_xlabel("Time")
ax1.set_ylabel("Valuation")
ax2.set_ylabel("Positive Tweets")

plt.legend()
plt.show()


# ### We notice that where there is a spike in Positive Tweets, there is spike in the valuation
# 
# ### Let us plot line graph to analyse effect of Negative Tweets & its effect on net gain/loss in valuation

# In[56]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.figure(figsize=(15, 6))
plt.title("Tesla: Effect of Negative Tweets on Value Gain/Loss")
# since we are plotting 2 line graphs with same X-axis
ax1 = plt.gca()
ax2 = plt.twinx()


ax1.plot(
    ovr_neg_tesla_df.date_str_x,
    ovr_neg_tesla_df.price_gain,
    color="b",
    label="Value",
)
ax2.plot(
    ovr_neg_tesla_df.date_str_x,
    ovr_neg_tesla_df.trend_score,
    color="r",
    label="Negative Tweets",
)

ax1.set_xlabel("Time")
ax1.set_ylabel("Value Gain/Loss")
ax2.set_ylabel("Negative Tweets")

plt.legend()
plt.show()


# ### We see that it becomes difficult to understand the effect of negative tweets on stock price. Hence, let us plot by taking the log values of "price_gain"

# In[57]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.figure(figsize=(15, 6))
plt.title("Tesla: Effect of Negative Tweets on Value Gain/Loss")
# since we are plotting 2 line graphs with same X-axis
ax1 = plt.gca()
ax2 = plt.twinx()


ax1.plot(
    ovr_neg_tesla_df.date_str_x,
    np.log(ovr_neg_tesla_df.price_gain),
    color="b",
    label="Value",
)
ax2.plot(
    ovr_neg_tesla_df.date_str_x,
    ovr_neg_tesla_df.trend_score,
    color="r",
    label="Negative Tweets",
)

ax1.set_xlabel("Time")
ax1.set_ylabel("Value Gain/Loss")
ax2.set_ylabel("Negative Tweets")

plt.legend()
plt.show()


# #### Taking logs seems to provide better viewing of the graph. Hence, we will take log values of 'price_gain' column while plotting

# ### From the above graph we see that where there is a spike in Negative Tweets, 
# ### the price gain is negative i.e. there is drop in valuation
# 
# ### We can conclude that Tesla's valuation is affected by Tweets

# ## Apple

# ### Let us analyse for Apple
# 
# ### We will analyse effect of Positive Tweets on Valuation

# In[58]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.figure(figsize=(15, 6))
plt.title("Apple: Effect of Positive Tweets on Valuation")
# since we are plotting 2 line graphs with same X-axis
ax1 = plt.gca()
ax2 = plt.twinx()


ax1.plot(
    ovr_pos_apple_df.date_str_x,
    ovr_pos_apple_df.total_valuation_EOD,
    color="b",
    label="Valuation",
)
ax2.plot(
    ovr_pos_apple_df.date_str_x,
    ovr_pos_apple_df.trend_score,
    color="g",
    label="Positive Tweets",
)

ax1.set_xlabel("Time")
ax1.set_ylabel("Valuation")
ax2.set_ylabel("Positive Tweets")

plt.legend()
plt.show()


# ### We notice that where there is a spike in Positive Tweets, there is spike in the valuation
# 
# ### Let us plot line graph to analyse effect of Negative Tweets & its effect on net gain/loss in valuation

# In[59]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.figure(figsize=(15, 6))
plt.title("Apple: Effect of Negative Tweets on Value Gain/Loss")
# since we are plotting 2 line graphs with same X-axis
ax1 = plt.gca()
ax2 = plt.twinx()


ax1.plot(
    ovr_neg_apple_df.date_str_x,
    np.log(ovr_neg_apple_df.price_gain),
    color="b",
    label="Value",
)
ax2.plot(
    ovr_neg_apple_df.date_str_x,
    ovr_neg_apple_df.trend_score,
    color="r",
    label="Negative Tweets",
)

ax1.set_xlabel("Time")
ax1.set_ylabel("Value Gain/Loss")
ax2.set_ylabel("Negative Tweets")

plt.legend()
plt.show()


# ### From the above graph we see that there is no proper correlation between negative tweets and drop in Apple share value
# ### We can conclude that Apple's valuation is affected only by Positive Tweets

# Amazon
# 
# 
# Let us analyse for Amazon
# 
# We will analyse effect of Positive Tweets on Valuation

# In[60]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.figure(figsize=(15, 6))
plt.title("Amazon: Effect of Positove Tweets on Valuation")
# since we are plotting 2 line graphs with same X-axis
ax1 = plt.gca()
ax2 = plt.twinx()


ax1.plot(
    ovr_pos_amazon_df.date_str_x,
    ovr_pos_amazon_df.total_valuation_EOD,
    color="b",
    label="Valuation",
)
ax2.plot(
    ovr_pos_amazon_df.date_str_x,
    ovr_pos_amazon_df.trend_score,
    color="g",
    label="Positive Tweets",
)

ax1.set_xlabel("Time")
ax1.set_ylabel("Valuation")
ax2.set_ylabel("Positive Tweets")

plt.legend()
plt.show()


# We notice that where there is a spike in Positive Tweets, there is spike in the valuation
# 
# Let us plot line graph to analyse effect of Negative Tweets & its effect on net gain/loss in valuation

# In[61]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.figure(figsize=(15, 6))
plt.title("Amazon: Effect of Negative Tweets on Value Gain/Loss")
# since we are plotting 2 line graphs with same X-axis
ax1 = plt.gca()
ax2 = plt.twinx()


ax1.plot(
    ovr_neg_amazon_df.date_str_x,
    np.log(ovr_neg_amazon_df.price_gain),
    color="y",
    label="Value",
)
ax2.plot(
    ovr_neg_amazon_df.date_str_x,
    ovr_neg_amazon_df.trend_score,
    color="r",
    label="Negative Tweets",
)

ax1.set_xlabel("Time")
ax1.set_ylabel("Valuation")
ax2.set_ylabel("Negative Tweets")

plt.legend()
plt.show()


# From the above graph we see that where there is a spike in Negative Tweets,
# 
# the price gain is negative i.e. there is drop in valuation
# 
# We can conclude that Amazon's valuation is affected by Tweets

# Microsoft
# 
# Let us analyse for Microsoft
# 
# We will analyse effect of Positive Tweets on Valuation

# In[62]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.figure(figsize=(15, 6))
plt.title("Microsoft: Effect of Positove Tweets on Valuation")
# since we are plotting 2 line graphs with same X-axis
ax1 = plt.gca()
ax2 = plt.twinx()


ax1.plot(
    ovr_pos_microsoft_df.date_str_x,
    ovr_pos_microsoft_df.total_valuation_EOD,
    color="b",
    label="Valuation",
)
ax2.plot(
    ovr_pos_microsoft_df.date_str_x,
    ovr_pos_microsoft_df.trend_score,
    color="g",
    label="Positive Tweets",
)

ax1.set_xlabel("Time")
ax1.set_ylabel("Valuation")
ax2.set_ylabel("Positive Tweets")

plt.legend()
plt.show()


# We notice that where there is a spike in Positive Tweets, there is spike in the valuation
# 
# Let us plot line graph to analyse effect of Negative Tweets & its effect on net gain/loss in valuation

# In[63]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.figure(figsize=(15, 6))
plt.title("Microsoft: Effect of Negative Tweets on Value Gain/Loss")
# since we are plotting 2 line graphs with same X-axis
ax1 = plt.gca()
ax2 = plt.twinx()


ax1.plot(
    ovr_neg_microsoft_df.date_str_x,
    np.log(ovr_neg_microsoft_df.price_gain),
    color="b",
    label="Value",
)
ax2.plot(
    ovr_neg_microsoft_df.date_str_x,
    ovr_neg_microsoft_df.trend_score,
    color="r",
    label="Negative Tweets",
)

ax1.set_xlabel("Time")
ax1.set_ylabel("Valuation")
ax2.set_ylabel("Negative Tweets")

plt.legend()
plt.show()


# From the above graph we see that generally where there is a spike in Negative Tweets,
# 
# the price gain is negative i.e. there is drop in valuation
# 
# We can conclude that Microsoft's valuation is affected by Tweets

# **Conclusion**
# 
# From analysing the given data, we conclude that:
# 
# 1)Positive Tweet Spikes (high trending) coincide with rise in stock value
# 
# 2)Negative Tweet Spikes (high trending) overall coincide with drop in stock value, however it is not as obvious as Positive Tweets
# 
# 3)(Naturally) there are also other factors contributing to rise/fall of stock values as we see no relation to rise/fall of stock value with low-medium trending tweets
