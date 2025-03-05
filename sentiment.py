import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
import re
from wordcloud import WordCloud
from gtts import gTTS
import base64
from sklearn.linear_model import LinearRegression
import pdfkit
import text2emotion as te

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Set page title and layout
st.set_page_config(page_title="Harare City Services Sentiment Analysis", layout="wide")

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Login form
def login():
    st.sidebar.header("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username == "HARARECITY" and password == "FORALLPEOPLE":  # Replace with your authentication logic
            st.session_state['logged_in'] = True
        else:
            st.sidebar.error("Invalid username or password")

# Logout button
def logout():
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False

# Show login form if not logged in
if not st.session_state['logged_in']:
    login()
    st.stop()  # Stop the app if the user is not logged in
else:
    logout()

# Title
st.title("Harare City Services Sentiment Analysis")
st.markdown("Monitor public sentiment about city services and identify areas of concern.")

# Dark Mode Toggle
dark_mode = st.sidebar.checkbox("Dark Mode")
if dark_mode:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Function to generate narration audio
def generate_narration(text, filename="narration.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename

# Function to autoplay audio
def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

# Narration script
narration_script = """
Welcome to the Harare City Services Sentiment Analysis app. 
This app helps you monitor public sentiment about city services and identify areas of concern.
You can view the public feedback dataset, analyze sentiment distribution, and explore insights.
Let's get started!
"""

# Generate and play narration
if st.button("Start Narration"):
    narration_file = generate_narration(narration_script)
    autoplay_audio(narration_file)

# Load sample dataset
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        data = {
            "Date": ["2023-10-01", "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05"],
            "Feedback": [
                "The water supply has been terrible this week.",
                "Waste management is improving, but more bins are needed.",
                "Transportation services are unreliable and expensive.",
                "The council is doing a great job with road repairs.",
                "No water for days, this is unacceptable!"
            ],
            "Service": ["Water Supply", "Waste Management", "Transportation", "Road Repairs", "Water Supply"]
        }
        df = pd.DataFrame(data)
    return df

# File uploader for admin users
st.sidebar.header("Admin Panel")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

# Load data
df = load_data(uploaded_file)

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df['Cleaned_Feedback'] = df['Feedback'].apply(preprocess_text)

# Perform sentiment analysis using TextBlob
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df['Sentiment'] = df['Cleaned_Feedback'].apply(get_sentiment)

# Display dataset
st.subheader("Public Feedback Dataset")
st.write(df)

# Narration for dataset
dataset_narration = """
Here is the public feedback dataset. It contains the date, feedback text, and the service category.
You can scroll through the data to see what people are saying about city services.
"""
if st.button("Explain Dataset"):
    narration_file = generate_narration(dataset_narration)
    autoplay_audio(narration_file)

# Sentiment distribution
st.subheader("Sentiment Distribution")
sentiment_counts = df['Sentiment'].value_counts()
st.bar_chart(sentiment_counts)

# Narration for sentiment distribution
sentiment_dist_narration = """
This chart shows the distribution of sentiment across all feedback. 
You can see how many feedback entries are positive, negative, or neutral.
"""
if st.button("Explain Sentiment Distribution"):
    narration_file = generate_narration(sentiment_dist_narration)
    autoplay_audio(narration_file)

# Sentiment by service
st.subheader("Sentiment by Service")
sentiment_service = df.groupby(['Service', 'Sentiment']).size().unstack()
st.bar_chart(sentiment_service)

# Narration for sentiment by service
sentiment_service_narration = """
This chart breaks down sentiment by service category. 
It helps you identify which services are receiving positive or negative feedback.
"""
if st.button("Explain Sentiment by Service"):
    narration_file = generate_narration(sentiment_service_narration)
    autoplay_audio(narration_file)

# Time trend of sentiment
st.subheader("Sentiment Over Time")
df['Date'] = pd.to_datetime(df['Date'])
time_sentiment = df.groupby([df['Date'].dt.date, 'Sentiment']).size().unstack()
st.line_chart(time_sentiment)

# Narration for sentiment over time
sentiment_time_narration = """
This line chart shows how sentiment has changed over time. 
You can track whether public sentiment is improving or worsening.
"""
if st.button("Explain Sentiment Over Time"):
    narration_file = generate_narration(sentiment_time_narration)
    autoplay_audio(narration_file)

# Word cloud for negative feedback
st.subheader("Word Cloud for Negative Feedback")
negative_feedback = " ".join(df[df['Sentiment'] == 'Negative']['Cleaned_Feedback'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_feedback)
st.image(wordcloud.to_array(), use_column_width=True)

# Narration for word cloud
wordcloud_narration = """
This word cloud visualizes the most common words in negative feedback. 
It helps you quickly identify recurring issues or complaints.
"""
if st.button("Explain Word Cloud"):
    narration_file = generate_narration(wordcloud_narration)
    autoplay_audio(narration_file)

# Key insights
st.subheader("Key Insights")
st.write("1. **Most Common Issues**: Identify the most frequently mentioned problems in negative feedback.")
st.write("2. **Service Performance**: Compare sentiment across different city services.")
st.write("3. **Trends Over Time**: Track how sentiment changes over time to measure the impact of council initiatives.")

# Narration for key insights
insights_narration = """
Here are some key insights from the analysis:
1. Identify the most common issues mentioned in negative feedback.
2. Compare sentiment across different city services.
3. Track how sentiment changes over time to measure the impact of council initiatives.
"""
if st.button("Explain Key Insights"):
    narration_file = generate_narration(insights_narration)
    autoplay_audio(narration_file)

# Predictive Sentiment Analysis
st.subheader("Predicted Sentiment Over Next 30 Days")
df['Days'] = (df['Date'] - df['Date'].min()).dt.days
model = LinearRegression()
model.fit(df[['Days']], df['Sentiment'].map({'Negative': -1, 'Neutral': 0, 'Positive': 1}))
future_days = 30
future_dates = pd.date_range(df['Date'].max(), periods=future_days, freq='D')
future_df = pd.DataFrame({'Date': future_dates})
future_df['Days'] = (future_df['Date'] - df['Date'].min()).dt.days
future_df['Predicted_Sentiment'] = model.predict(future_df[['Days']])
st.line_chart(future_df.set_index('Date')['Predicted_Sentiment'])

# Narration for predicted sentiment
predicted_sentiment_narration = """
This graph shows the predicted sentiment trend for the next 30 days. 
It uses historical data to forecast whether public sentiment is likely to improve or worsen.
"""
if st.button("Explain Predicted Sentiment"):
    narration_file = generate_narration(predicted_sentiment_narration)
    autoplay_audio(narration_file)

# Filter & Search
st.sidebar.header("Filter & Search")
sentiment_filter = st.sidebar.multiselect("Filter by Sentiment", options=df['Sentiment'].unique(), default=df['Sentiment'].unique())
date_range = st.sidebar.date_input("Filter by Date Range", [df['Date'].min().date(), df['Date'].max().date()])
keyword_filter = st.sidebar.text_input("Filter by Keywords")

# Apply filters
filtered_df = df[
    (df['Sentiment'].isin(sentiment_filter)) &
    (df['Date'].dt.date >= date_range[0]) &
    (df['Date'].dt.date <= date_range[1]) &
    (df['Cleaned_Feedback'].str.contains(keyword_filter, case=False))
]

# Display filtered results
st.subheader("Filtered Results")
st.write(filtered_df)

# Customizable Reports
st.subheader("Download Reports")
if st.button("Generate PDF Report"):
    pdf = pdfkit.from_string(filtered_df.to_html(), False)
    st.download_button(label="Download PDF", data=pdf, file_name="report.pdf", mime="application/pdf")
if st.button("Generate CSV Report"):
    csv = filtered_df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name="report.csv", mime="text/csv")

# Aspect-Based Sentiment Analysis
st.subheader("Aspect-Based Sentiment Analysis")
aspects = ["service", "quality", "price", "delivery"]
for aspect in aspects:
    aspect_df = df[df['Cleaned_Feedback'].str.contains(aspect, case=False)]
    if not aspect_df.empty:
        st.write(f"Sentiment for {aspect.capitalize()}")
        st.bar_chart(aspect_df['Sentiment'].value_counts())

# Narration for Aspect-Based Sentiment Analysis
aspect_narration = """
This section breaks down sentiment by specific aspects such as service, quality, price, and delivery.
It helps you understand how people feel about each aspect of city services.
"""
if st.button("Explain Aspect-Based Sentiment Analysis"):
    narration_file = generate_narration(aspect_narration)
    autoplay_audio(narration_file)

# Emotion Detection
st.subheader("Emotion Detection")
try:
    df['Emotions'] = df['Cleaned_Feedback'].apply(te.get_emotion)
    st.write(df[['Cleaned_Feedback', 'Emotions']])
except AttributeError as e:
    st.error(f"Error in emotion detection: {e}")
    st.write("Please ensure the `emoji` library is version 1.7.0. Run: `pip install emoji==1.7.0`")

# Narration for Emotion Detection
emotion_narration = """
This section detects emotions such as joy, anger, sadness, and surprise in the feedback.
It provides a deeper understanding of how people feel beyond just positive or negative sentiment.
"""
if st.button("Explain Emotion Detection"):
    narration_file = generate_narration(emotion_narration)
    autoplay_audio(narration_file)

# Comparison Mode
uploaded_file2 = st.sidebar.file_uploader("Upload a second dataset for comparison (CSV)", type=["csv"])
if uploaded_file2:
    df2 = load_data(uploaded_file2)
    df2['Sentiment'] = df2['Cleaned_Feedback'].apply(get_sentiment)
    st.subheader("Comparison of Sentiment Distribution")
    st.bar_chart(df['Sentiment'].value_counts())
    st.bar_chart(df2['Sentiment'].value_counts())

# Footer
st.markdown("---")
st.markdown("Developed for Harare City Council to improve public services and decision-making.")