# Studying the Correlation between Tweet Sentiment and Stock Prices

## Overview
The project aims to investigate the impact of social media sentiment on stock prices using Twitter data and sentiment analysis. Twitter provides real-time information about user reactions and opinions towards various topics, including stocks and companies. By analyzing these tweets using sentiment analysis techniques, we can classify them as positive, negative, or neutral and determine the overall sentiment towards a particular stock or company.

## Technologies Used
- Python
- Jupyter Notebook
- Pandas
- Matplotlib
- Plotly
- NLTK (Natural Language Toolkit)
- VADER (Valence Aware Dictionary and Sentiment Reasoner)
- TextBlob

## Project Description
The project involves the following steps:
1. Loading and cleaning the data: Loading the Twitter data and stock price data, cleaning the datasets, and merging them for analysis.
2. Sentiment analysis: Performing sentiment analysis on the tweets using both VADER and TextBlob libraries to classify them as positive, negative, or neutral.
3. Data analysis: Analyzing the correlation between tweet sentiment and stock prices for different companies.
4. Visualization: Visualizing the data using line plots to understand the impact of positive and negative tweets on stock prices.

## How to Use
To clone and run the code locally, follow these steps:
1. Clone the repository: `git clone https://github.com/brijeshverma0007/tweet-sentiment-analysis.git`
2. Navigate to the project directory: `cd tweet-sentiment-analysis`
3. Install the required dependencies manually using pip:
   - pandas: `pip install pandas` ([documentation](https://pandas.pydata.org/docs/))
   - matplotlib: `pip install matplotlib` ([documentation](https://matplotlib.org/stable/contents.html))
   - numpy: `pip install numpy` ([documentation](https://numpy.org/doc/stable/))
   - plotly: `pip install plotly` ([documentation](https://plotly.com/python/getting-started/))
   - textblob: `pip install textblob` ([documentation](https://textblob.readthedocs.io/en/dev/))
   - seaborn: `pip install seaborn` ([documentation](https://seaborn.pydata.org/))
   - twython: `pip install twython` ([documentation](https://twython.readthedocs.io/en/latest/))
   - varname: `pip install varname` ([documentation](https://pypi.org/project/varname/))

## **Running the code**

### 1. Running on Local Machine with Jupyter Notebook

1. Download the five CSV files (Company.csv, Company_Tweet.csv, CompanyValue.csv, Tweet.csv, sample_train_raw_tweet_db.csv) from ([this Google Drive link](https://drive.google.com/drive/folders/1K1ZHSeN37nDNc8HL8vU9xAqQfL-LEnCC?usp=sharing)).
2. Place the CSV files in the project directory.
3. Ensure Python and pip are installed on your machine. Pip is required to download necessary packages.
4. Open the Jupyter Notebook file `tweet_sentiment_analysis.ipynb`.
5. Comment out the code for mounting the drive to Google Colab, as it's not needed for local execution.
6. Run the code snippets one by one or execute the entire notebook.
7. For sentiment analysis accuracy testing, use the provided sample_train_raw_tweet_db.csv file, which contains manually labeled sentiments for tweets.

### 2. Running with Google Colab

1. Upload the Jupyter Notebook file and the five CSV files to Google Drive.
2. Alternatively, add shortcuts to these files directly from ([this Google Drive link](https://drive.google.com/drive/folders/1K1ZHSeN37nDNc8HL8vU9xAqQfL-LEnCC?usp=sharing)) to avoid re-uploading.
3. Open the Jupyter Notebook file in Google Colab.
4. Ensure all required packages are installed; they are included in the notebook with `pip install` commands.
5. Mount the drive to the Colab notebook using the provided command in the notebook.
6. Run the code snippets individually or execute the entire notebook.
7. Use the sample_train_raw_tweet_db.csv file for sentiment analysis accuracy evaluation.

*By following these steps, you can run the sentiment analysis project either locally with Jupyter Notebook or on Google Colab.*

### 3. Alternatively, you can run the code using the Python terminal:
1. Open a terminal window.
2. Navigate to the project directory: `cd tweet-sentiment-analysis`
3. Run the code with python command: `python tweet_sentiment_analysis_py.py`


## Conclusion
The project provides valuable insights into the impact of social media sentiment on stock prices. By leveraging sentiment analysis techniques, investors and traders can gain a better understanding of the sentiment of social media users towards a particular stock or company. This information can help them make informed decisions and potentially gain a competitive advantage in the market.
