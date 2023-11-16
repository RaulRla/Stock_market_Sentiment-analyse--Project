import pandas as pd
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
import requests
import plotly.graph_objs as go
import datetime as dt
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
import matplotlib.pyplot as plt

# Function to perform sentiment analysis
def perform_sentiment_analysis(news_df):
    positive = 0
    negative = 0
    neutral = 0
    news_list = []
    neutral_list = []
    negative_list = []
    positive_list = []

    for news in news_df['Summary']:
        news_list.append(news)
        analyzer = SentimentIntensityAnalyzer().polarity_scores(news)
        neg = analyzer['neg']
        neu = analyzer['neu']
        pos = analyzer['pos']
        comp = analyzer['compound']

        if neg > pos:
            negative_list.append(news)
            negative += 1
        elif pos > neg:
            positive_list.append(news)
            positive += 1
        elif pos == neg:
            neutral_list.append(news)
            neutral += 1

    positive_percentage = 100 * positive / len(news_df)
    negative_percentage = 100 * negative / len(news_df)
    neutral_percentage = 100 * neutral / len(news_df)

    sentiment_result = {
        'Positive': positive_percentage,
        'Neutral': neutral_percentage,
        'Negative': negative_percentage
    }

    return sentiment_result, news_list, neutral_list, negative_list, positive_list

# Function to generate word cloud
def generate_word_cloud(text):
    stopwords = set(STOPWORDS)
    all_words = ' '.join([news for news in text])
    word_cloud = WordCloud(
        background_color='black', width=800, height=400, stopwords=stopwords,
        min_font_size=20, max_font_size=150, colormap='ocean'
    ).generate(all_words)

    return word_cloud

# Streamlit app
st.title("Nalises")

ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Select Start Date", value=None)
end_date = st.sidebar.date_input("Select End Date", value=None)

news_input = st.text_area("Input Company Name")
search_news_button = st.button("Search News")

news_df = pd.DataFrame(columns=["Summary"])
fetching_data = st.empty()

if ticker:
    if start_date and end_date:
        df = yf.download(ticker, start=start_date, end=end_date)
    else:
        df = yf.download(ticker)

    st.write("Stock Data:")
    st.write(df)

if not df.empty:
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    st.write("Candlestick Chart:")
    st.plotly_chart(fig)

if search_news_button:
    now = dt.date.today()
    now = now.strftime('%m-%d-%Y')
    yesterday = dt.date.today() - dt.timedelta(days=1)
    yesterday = yesterday.strftime('%m-%d-%Y')

    nltk.download('punkt')
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
    config = Config()
    config.browser_user_agent = user_agent
    config.request_timeout = 10

    company_name = news_input

    if company_name != '':
        st.write(f'Searching for and analyzing {company_name}, Please be patient, it might take a while...')

        googlenews = GoogleNews(start=yesterday, end=now)
        googlenews.search(company_name)
        result = googlenews.result()
        df = pd.DataFrame(result)

        news_list = []
        news_counter = 0

        fetching_data.text("Fetching news data...")

        for i in df.index:
            if news_counter >= 20:
                break

            try:
                article = Article(df['link'][i], config=config)
                article.download()
                article.parse()
                article.nlp()

                news_list.append(article.text)
                news_counter += 1
            except Exception as e:
                print(f"Error while collecting news: {str(e)}")
                continue

        if not any(news_list):
            st.write("No news articles found for the given ticker.")
        else:
            new_news_df = pd.DataFrame({'Summary': news_list})
            news_df = pd.concat([news_df, new_news_df], ignore_index=True)
            fetching_data.empty()
            st.write("News Data:")
            st.write(news_df)

            # Perform sentiment analysis after collecting news
            sentiment_result, _, _, _, _ = perform_sentiment_analysis(news_df)
            st.write("Sentiment Analysis Result:")
            st.write(sentiment_result)

if not news_df.empty:
    word_cloud = generate_word_cloud(news_df['Summary'].values)
    st.write("Word Cloud for News Data")
    st.image(word_cloud.to_image())

# Create and display the sentiment analysis results

if not news_df.empty:
    sentiment_result, news_list, neutral_list, negative_list, positive_list = perform_sentiment_analysis(news_df)
    st.write("Sentiment Analysis Result:")
    st.write(sentiment_result)

    st.write("Neutral Sentiment News:")
    st.write(neutral_list)

    st.write("Negative Sentiment News:")
    st.write(negative_list)

    st.write("Positive Sentiment News:")
    st.write(positive_list)

st.pyplot(plt)  # Display any Matplotlib plots
