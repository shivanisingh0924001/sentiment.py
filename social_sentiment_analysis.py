import json
import pandas as pd
from textblob import TextBlob
import mysql.connector
import streamlit as st
import sys
import io
import matplotlib.pyplot as plt # type: ignore
import numpy as np

# --- Streamlit Page Config ---
st.set_page_config(page_title="Social Media Sentiment Dashboard", layout="wide")
st.title("📊 Social Media Sentiment Analysis")

# --- Server Health Check ---
if st.runtime.exists():
    st.sidebar.success("✅ Server connection active")
else:
    st.sidebar.error("❌ Server connection lost! Please refresh the page")

# --- Load and Normalize JSON Data ---
try:
    with open('social_data.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Normalize engagement metrics across platforms
    for item in data:
        # Handle inconsistent keys for shares/retweets
        if 'retweets' not in item:
            if 'shares' in item:
                item['retweets'] = item['shares']
            elif 'instagram shares' in item:
                item['retweets'] = item['instagram shares']
            elif 'facebook shares' in item:
                item['retweets'] = item['facebook shares']
            else:
                item['retweets'] = 0
        
        # Ensure all items have consistent keys
        item.setdefault('likes', 0)
        item.setdefault('retweets', 0)
        item.setdefault('location', 'Unknown')
        item.setdefault('platform', 'Unknown')
    
    st.success("✅ JSON data loaded and normalized successfully!")
    
    # Debug: Show first 3 entries
    with st.expander("Show first 3 JSON entries"):
        st.json(data[:3])
        
except FileNotFoundError:
    st.error("❌ File 'social_data.json' not found. Please check the file path.")
    st.stop()
except json.JSONDecodeError:
    st.error("❌ JSON decoding error. Please check the file format.")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected error loading JSON: {str(e)}")
    st.stop()

# --- Convert to DataFrame ---
df = pd.DataFrame(data)

if df.empty:
    st.warning("⚠️ Loaded data is empty. Please check your JSON file.")
    st.stop()

# --- Handle timestamps with fallback ---
if 'timestamp' in df.columns:
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        invalid_count = df['timestamp'].isna().sum()
        if invalid_count > 0:
            st.warning(f"⚠️ {invalid_count} invalid timestamps found. Using current time as fallback.")
            df.loc[df['timestamp'].isna(), 'timestamp'] = pd.Timestamp.now(tz='UTC')
    except Exception as e:
        st.error(f"Timestamp conversion failed: {str(e)}")
        df['timestamp'] = pd.Timestamp.now(tz='UTC')
else:
    st.warning("⚠️ 'timestamp' column not found - using current time")
    df['timestamp'] = pd.Timestamp.now(tz='UTC')

# --- Sentiment Analysis with robust error handling ---
def analyze_sentiment(text):
    try:
        if not isinstance(text, str) or text.strip() == "":
            return 0.0, "Neutral"
        
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        
        if polarity > 0.1:
            return polarity, "Positive"
        elif polarity < -0.1:
            return polarity, "Negative"
        else:
            return polarity, "Neutral"
    except Exception as e:
        return 0.0, "Neutral"

# Apply sentiment analysis safely
df[['Sentiment_Score', 'Sentiment']] = df['text'].apply(
    lambda x: pd.Series(analyze_sentiment(x)))
    
# --- MySQL Database Handling ---
db_status = "Database not attempted"
inserted_count = 0

try:
    with st.spinner("🔌 Connecting to MySQL database..."):
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='12345',
            database='social_posts',
            connection_timeout=5
        )
        cursor = conn.cursor()
        db_status = "✅ Connected to MySQL database"

    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS social_posts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id VARCHAR(100),
            username VARCHAR(100),
            text TEXT,
            sentiment_score FLOAT,
            sentiment VARCHAR(20),
            timestamp DATETIME,
            platform VARCHAR(50),
            location VARCHAR(100),
            likes INT,
            retweets INT
        )
    """)
    db_status += " | Table created/verified"

    # Prepare data for batch insert
    records = []
    for _, row in df.iterrows():
        # Safely handle engagement metrics
        likes = row.get('likes', 0)
        retweets = row.get('retweets', 0)
        
        # Convert to integers safely
        try:
            likes = int(likes) if not pd.isna(likes) else 0
        except (ValueError, TypeError):
            likes = 0
            
        try:
            retweets = int(retweets) if not pd.isna(retweets) else 0
        except (ValueError, TypeError):
            retweets = 0
        
        records.append((
            row.get('user_id'),
            row.get('username'),
            row.get('text'),
            row['Sentiment_Score'],
            row['Sentiment'],
            row.get('timestamp'),
            row.get('platform'),
            row.get('location'),
            likes,
            retweets
        ))

    # Batch insert
    with st.spinner(f"💾 Inserting {len(records)} records into database..."):
        cursor.executemany("""
            INSERT INTO social_posts 
            (user_id, username, text, sentiment_score, sentiment, timestamp, platform, location, likes, retweets)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, records)
        inserted_count = cursor.rowcount
        conn.commit()
        db_status += f" | {inserted_count} records inserted"

except mysql.connector.Error as err:
    db_status = f"❌ MySQL Error: {err}"
    
    # Specific troubleshooting guidance
    if "1045" in str(err):
        st.error("Invalid database credentials. Check username/password")
    elif "2003" in str(err):
        st.error("Cannot connect to MySQL server. Ensure it's running with: sudo service mysql start")
    elif "1049" in str(err):
        st.error("Database 'social_posts' doesn't exist. Create it with:")
        st.code("""
        mysql -u root -p12345 -e "CREATE DATABASE social_posts;"
        """)
    else:
        st.error(f"Unhandled MySQL error: {err}")
        
except Exception as e:
    db_status = f"❌ Unexpected Error: {e}"
    st.error(f"Database operation failed: {str(e)}")

finally:
    if 'conn' in locals() and conn.is_connected():
        cursor.close()
        conn.close()
        st.sidebar.success("MySQL connection closed")

# Show database status
if db_status.startswith("✅"):
    st.success(db_status)
else:
    st.error(db_status)

# --- Filtering ---
if 'platform' in df.columns and not df['platform'].empty:
    platforms = ['All'] + sorted(df['platform'].dropna().unique().tolist())
    selected_platform = st.sidebar.selectbox("🔍 Filter by Platform", platforms)
    filtered_df = df if selected_platform == "All" else df[df['platform'] == selected_platform]
else:
    st.warning("⚠️ Platform data not available for filtering")
    filtered_df = df

# --- Sentiment Distribution (Matplotlib version) ---
try:
    st.subheader("📊 Sentiment Distribution")
    
    if 'Sentiment' in filtered_df.columns:
        sentiment_counts = filtered_df['Sentiment'].value_counts()
        
        if not sentiment_counts.empty:
            # Create a clean matplotlib figure
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}
            sentiment_counts.plot(kind='bar', 
                                 color=[colors.get(x, 'gray') for x in sentiment_counts.index],
                                 ax=ax)
            
            ax.set_title('Sentiment Distribution')
            ax.set_ylabel('Count')
            ax.set_xlabel('Sentiment')
            plt.xticks(rotation=0)
            st.pyplot(fig)
        else:
            st.warning("No sentiment data available for visualization")
    else:
        st.warning("Sentiment data not found in DataFrame")
except Exception as e:
    st.error(f"Error displaying sentiment distribution: {str(e)}")

# --- Sentiment Over Time ---
try:
    st.subheader("📅 Sentiment Trend Over Time")
    if 'timestamp' in filtered_df.columns and 'Sentiment_Score' in filtered_df.columns:
        # Create a copy to avoid SettingWithCopyWarning
        time_df = filtered_df.copy()
        time_df['Sentiment_Score'] = pd.to_numeric(time_df['Sentiment_Score'], errors='coerce')
        time_df = time_df.set_index('timestamp')
        
        # Resample and calculate daily average
        daily_avg = time_df['Sentiment_Score'].resample('D').mean().fillna(0)
        
        if not daily_avg.empty:
            # Use matplotlib for plotting
            fig, ax = plt.subplots(figsize=(10, 4))
            daily_avg.plot(ax=ax, color='purple', marker='o')
            ax.set_title('Daily Average Sentiment Score')
            ax.set_ylabel('Sentiment Score')
            ax.set_xlabel('Date')
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
        else:
            st.warning("No valid data for trend visualization")
    else:
        st.warning("Timestamp or Sentiment_Score data missing for trend chart")
except Exception as e:
    st.error(f"Error generating trend chart: {str(e)}")

# --- Data Preview ---
st.subheader("📄 Filtered Data Preview")
st.dataframe(filtered_df.head(20))

# --- CSV Export ---
try:
    # Create a copy for export to avoid modifying original
    export_df = filtered_df.copy()
    
    # Handle datetime conversion for export
    if 'timestamp' in export_df.columns:
        export_df['timestamp'] = export_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Download CSV",
        data=csv,
        file_name="social_media_sentiment.csv",
        mime="text/csv"
    )
    st.success("CSV ready for download!")
except Exception as e:
    st.error(f"Error generating CSV: {str(e)}")

# --- Debug Info ---
st.sidebar.subheader("ℹ️ System Information")
st.sidebar.write(f"Python: {sys.version.split()[0]}")
st.sidebar.write(f"Streamlit: {st.__version__}")
st.sidebar.write(f"Pandas: {pd.__version__}")
st.sidebar.write(f"MySQL Connector: {mysql.connector.__version__}")

# --- Database Troubleshooting Guide ---
with st.sidebar.expander("⚠️ MySQL Troubleshooting"):
    st.markdown("""
    **Common fixes:**
    1. Start MySQL service:
        ```bash
        sudo service mysql start
        ```
    2. Create database manually:
        ```bash
        mysql -u root -p12345 -e "CREATE DATABASE social_posts;"
        ```
    3. Verify table structure:
        ```sql
        USE social_posts;
        DESCRIBE social_posts;
        ```
    4. Test connection:
        ```bash
        mysql -u root -p12345 -h localhost social_posts
        ```
    """)

# --- Data Quality Report ---
with st.expander("🔍 Data Quality Report"):
    st.subheader("Missing Values")
    missing_data = filtered_df.isna().sum()
    st.bar_chart(missing_data)
    
    st.subheader("Data Types")
    st.write(filtered_df.dtypes)
    
    if 'text' in filtered_df.columns:
        st.subheader("Text Length Distribution")
        filtered_df['text_length'] = filtered_df['text'].apply(lambda x: len(str(x)))
        st.bar_chart(filtered_df['text_length'].value_counts().head(20))
    
    st.subheader("Engagement Metrics Summary")
    st.write(filtered_df[['likes', 'retweets']].describe())