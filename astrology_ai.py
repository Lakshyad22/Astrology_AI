from skyfield.api import load, Topos
import pandas as pd
import streamlit as st
from datetime import datetime
import pytz
from geopy.geocoders import Nominatim
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px
import random  # Import random module

planets = load('de421.bsp')
earth = planets['earth']
ts = load.timescale()

st.set_page_config(layout="wide")
st.title("🌌 AI-Based Astrology Personality Profiler")
st.markdown("#### Enter your birth details for personality insights based on astronomical planetary positions.")

geolocator = Nominatim(user_agent="astro_app")  # Initialize geolocator

# Load data from CSV file
@st.cache_data  # Cache the data loading
def load_astrology_data():
    return pd.read_csv("astrology_data.csv")

astro_data = load_astrology_data()

with st.form("birth_form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Your Name", "Lakshya")
        date_input = st.date_input("Date of Birth", value=datetime(2000, 1, 1))
        time_input = st.time_input("Time of Birth")
        birth_city = st.text_input("City of Birth", "Jodhpur")

    with col2:
        timezone = st.selectbox("Timezone", pytz.all_timezones, index=pytz.all_timezones.index("Asia/Kolkata"))
        submit = st.form_submit_button("Generate Personality Chart")

if submit:
    try:
        location = geolocator.geocode(birth_city)
        if location:
            lat, lon = location.latitude, location.longitude
        else:
            st.error(f"Could not find coordinates for {birth_city}. Please check the city name.")
            st.stop()

        local_dt = pytz.timezone(timezone).localize(datetime.combine(date_input, time_input))
        utc_dt = local_dt.astimezone(pytz.utc)
        t = ts.utc(utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour, utc_dt.minute, utc_dt.second)

        # Construct the observer object AFTER the time is defined
        # Modification for older Skyfield: REMOVE t=ts
        observer = earth + Topos(latitude_degrees=lat, longitude_degrees=lon)

        planets_list = ['MERCURY BARYCENTER', 'VENUS BARYCENTER', 'MARS BARYCENTER', 'JUPITER BARYCENTER', 'SATURN BARYCENTER', 'URANUS BARYCENTER', 'NEPTUNE BARYCENTER', 'PLUTO BARYCENTER', 'MOON', 'SUN']
        positions = []

        # Dictionary to store planetary positions
        planetary_positions = {}

        # Iterate through the planets and calculate their positions
        for planet in planets_list:
            if planet in ('MOON', 'SUN'):
                ast_obj = planets[planet]
                ast = observer.at(t).observe(ast_obj)
                alt, az, distance = ast.apparent().altaz()
                ra, dec, dist = ast.apparent().radec()
                positions.append(f"{planet}: RA {ra.hours:.2f}h, DEC {dec.degrees:.2f}°, ALT {alt.degrees:.2f}°")
                planetary_positions[planet] = f"RA {ra.hours:.2f}h, DEC {dec.degrees:.2f}°, ALT {alt.degrees:.2f}°"
            else:
                ast_obj = planets[planet]
                ast = observer.at(t).observe(ast_obj)  # First observe with .at(t)
                alt, az, distance = ast.apparent().altaz()
                ra, dec, dist = ast.apparent().radec()
                positions.append(f"{planet}: RA {ra.hours:.2f}h, DEC {dec.degrees:.2f}°, ALT {alt.degrees:.2f}°")
                planetary_positions[planet] = f"RA {ra.hours:.2f}h, DEC {dec.degrees:.2f}°, ALT {alt.degrees:.2f}°"

        st.subheader(f"🔭 Planetary Positions for {name} (UTC: {utc_dt.strftime('%Y-%m-%d %H:%M:%S')})")
        st.markdown("#### Location: {}, Latitude: {}, Longitude: {}".format(birth_city, lat, lon))
        st.code("\n".join(positions))

        # --- NEW: Select personality profiles from CSV ---
        selected_profiles = {}
        for planet in planets_list:
            planet_data = astro_data[astro_data['planet'] == planet]
            if not planet_data.empty:
                # Choose a random profile
                selected_profile = planet_data.sample(n=1).iloc[0]  # Get the first row of the sample
                selected_profiles[planet] = selected_profile['description']
            else:
                selected_profiles[planet] = "N/A"  # Handle missing data

        st.subheader("🧬 AI-Inferred Personality Traits")
        trait_summary = "\n".join([f"- **{planet}** → {selected_profiles.get(planet, 'N/A')}" for planet in planetary_positions.keys()])
        st.markdown(trait_summary)

        # Data for clustering (using the selected profiles)
        sample_texts = list(selected_profiles.values())  # Use descriptions from selected profiles
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sample_texts)

        # Check if the number of samples is greater than the number of clusters
        n_samples = X.shape[0]

        if n_samples > 1:
            # Apply K-Means clustering
            kmeans = KMeans(n_clusters=min(3, n_samples), random_state=42, n_init = 'auto')  # Ensure n_clusters <= n_samples
            labels = kmeans.fit_predict(X)

            # Apply TSNE for dimensionality reduction (only if n_samples > 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(2, n_samples - 1))  # Ensure perplexity < n_samples
            X_embedded = tsne.fit_transform(X.toarray())

            # Create a DataFrame for visualization
            df_viz = pd.DataFrame(X_embedded, columns=['x', 'y'])
            df_viz['label'] = labels
            df_viz['planet'] = list(selected_profiles.keys())  # Use keys from selected_profiles

            # Create a scatter plot using Plotly Express
            fig = px.scatter(df_viz, x='x', y='y', color='label', text='planet',
                             title="🌌 Personality Clusters Based on Planetary Traits")

            # Display the plot using Streamlit
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data to perform clustering and visualization.")

        # --- ADDED SECTION:  Data Meaning & Learning ---
        st.subheader("🤔 Interpreting the Results")
        st.markdown("""
        The planetary positions at your birth time are calculated with high accuracy using established astronomical models.  However, the personality traits associated with these positions are based on traditional astrological beliefs.

        **What you might learn from this:**

        *   **Self-Reflection:** The descriptions can serve as prompts for self-reflection.  Do any of the described traits resonate with you?  Why or why not?
        *   **Areas of Potential:**  The profiles might highlight areas where you could focus your energy or develop your skills.  For example, if Mercury is prominent, communication skills might be a strength to cultivate.
        *   **Understanding Different Perspectives:**  Exploring astrological interpretations can offer insights into how different cultures and belief systems view personality and destiny.
        *   **Patterns:** The AI clustering shows which astrological interpretations are similar, according to textual analysis.
        """)

        # --- ADDED SECTION:  Disclaimer ---
        st.subheader("⚠️ Important Disclaimer")
        st.info("""
        This application provides astronomical data and astrological interpretations for informational and entertainment purposes only. **The personality traits described are based on traditional astrological beliefs and have not been scientifically validated.**  This information should not be used as a substitute for professional advice (e.g., psychological, medical, or financial).  Use this app responsibly and with an understanding that the results are based on belief, not empirical evidence.  We accurately calculate planetary positions, but the interpretations are subjective.
        """)


        st.markdown("---")
        st.markdown("🔧 Developed by **Lakshya Dulani** ")

    except Exception as e:
        st.error(f"An error occurred: {e}")
