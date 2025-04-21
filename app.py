import streamlit as st
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from transformers import pipeline
import plotly.express as px
from dotenv import load_dotenv
import os
load_dotenv()

import re

def extract_video_id(url):
    """
    Extracts video ID from standard YouTube URLs like:
    - https://www.youtube.com/watch?v=dQw4w9WgXcQ
    - https://youtu.be/dQw4w9WgXcQ
    - https://youtube.com/shorts/dQw4w9WgXcQ
    """
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*?$"  # watch?v= or /dQw4w9WgXcQ
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# Emotion model
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)

emotion_model = load_emotion_model()

def analyze_emotion(text):
    try:
        output = emotion_model(text[:512])
        scores = output[0]
        top = max(scores, key=lambda x: x['score'])
        return top['label'], top['score']
    except:
        return 'neutral', 0.0

def get_video_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    df = pd.DataFrame(transcript)
    df['minute'] = (df['start'] // 30).astype(int)
    return df

def extract_timestamps(comment):
    return re.findall(r'\b(\d{1,2}:\d{2})\b', comment)

def parse_timestamp(ts):
    parts = ts.split(":")
    return int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 else None

def get_comments(api_key, video_id, max_results=500):
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None
    while len(comments) < max_results:
        request = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=100,
            textFormat="plainText", pageToken=next_page_token
        )
        response = request.execute()
        for item in response['items']:
            text = item['snippet']['topLevelComment']['snippet']['textDisplay']
            timestamps = extract_timestamps(text)
            for ts in timestamps:
                sec = parse_timestamp(ts)
                if sec is not None:
                    comments.append({'timestamp': sec, 'text': text})
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return pd.DataFrame(comments)

def process_features_and_labels(transcript_df, comments_df, n_segments):
    comments_df['minute'] = (comments_df['timestamp'] // 30).astype(int)
    comments_df['sentiment'] = comments_df['text'].apply(lambda t: TextBlob(t).sentiment.polarity)
    comments_df[['emotion', 'emotion_score']] = comments_df['text'].apply(lambda t: pd.Series(analyze_emotion(t)))

    features = comments_df.groupby('minute').agg({
        'text': 'count',
        'sentiment': 'mean',
        'emotion_score': 'mean'
    }).rename(columns={'text': 'comment_count'}).reset_index()

    transcript_df['keywords'] = transcript_df['text'].str.lower().apply(lambda x: any(k in x for k in ['surprise', 'love', 'amazing', 'winner', 'crazy']))
    keyword_flag = transcript_df.groupby('minute')['keywords'].any().reset_index(name='has_keywords')

    all_minutes = pd.DataFrame({'minute': range(n_segments)})
    features = pd.merge(all_minutes, features, on='minute', how='left').fillna(0)
    features = pd.merge(features, keyword_flag, on='minute', how='left').fillna(False)
    features['has_keywords'] = features['has_keywords'].astype(int)

    z = (features['comment_count'] - features['comment_count'].mean()) / features['comment_count'].std()
    features['like_spike'] = (z > 1.0).astype(int)

    return features

def chunk_to_timestamp(chunk_index, chunk_duration=30):
    total_seconds = chunk_index * chunk_duration
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{int(minutes):02}:{int(seconds):02}"

# --- Streamlit UI ---
st.title("YouTube Viewer Engagement Analyzer")
api_key = os.getenv("YOUTUBE_API_KEY")
video_url = st.text_input("Paste YouTube Video Link")
video_id = extract_video_id(video_url)

if video_url and not video_id:
    st.warning("Couldn't extract video ID. Please check the URL.")

if api_key and video_id:
    with st.spinner("Fetching transcript and comments..."):
        transcript_df = get_video_transcript(video_id)
        comments_df = get_comments(api_key, video_id)
        n_segments = transcript_df['minute'].max() + 1
        features = process_features_and_labels(transcript_df, comments_df, n_segments)

    # Clustering
    X = features[['comment_count', 'sentiment', 'emotion_score', 'has_keywords']]
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    features['cluster'] = kmeans.fit_predict(X_scaled)

    cluster_labels = {0: "Low Engagement", 1: "Emotionally Mixed", 2: "Comment Spike"}
    features['cluster_label'] = features['cluster'].map(cluster_labels)
    unique_labels = features['cluster_label'].unique()
    color_map = {
        label: color for label, color in zip(
            unique_labels,
            ["#1f77b4", "#d62728", "#2ca02c"]  # blue, red, green
        )
    }


    features = features.drop_duplicates(subset='minute').copy()
    features = features.sort_values(by='minute').copy()
    features['timestamp'] = features['minute'].apply(chunk_to_timestamp)

    # Plot
    fig = px.bar(
        features,
        x='minute',
        y='comment_count',
        color='cluster_label',
        color_discrete_map=color_map,
        hover_data=['timestamp'],
        title='Viewer Engagement Clusters Across Video Timeline',
        labels={'minute': 'Time (30-sec chunks)', 'comment_count': 'Comments'}
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis=dict(
            tickmode='array',
            tickvals=features['minute'],
            ticktext=features['timestamp']
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    with st.expander("🔍 Raw Data"):
        st.dataframe(features)
