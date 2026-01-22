# ===================== app.py (updated) =====================
import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
from deep_translator import GoogleTranslator
from datetime import datetime
from shapely.geometry import Polygon
import joblib
import numpy as np
import json
from pathlib import Path
from PIL import Image
import io
import base64

# Optional: XGBoost (recommendation system)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# ------------------------
# CONFIG
# ------------------------
st.set_page_config(
    page_title="🌾 Yield Sense",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_KEY = "f5d74ad9384758204064758b2e9ba8a5"

# ------------------------
# TRANSLATION
# ------------------------
def t(text, target_lang):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except:
        return text

# ------------------------
# LANGUAGE OPTIONS
# ------------------------
language_options = {
    "en": "English",
    "hi": "हिन्दी (Hindi)",
    "bn": "বাংলা (Bengali)",
    "mr": "मराठी (Marathi)",
    "pa": "ਪੰਜਾਬੀ (Punjabi/Gurmukhi)",
    "gu": "ગુજરાતી (Gujarati)",
    "kn": "ಕನ್ನಡ (Kannada)",
    "ml": "മലയാളം (Malayalam)",
    "ta": "தமிழ் (Tamil)",
    "te": "తెలుగు (Telugu)",
    "ur": "اردو (Urdu)"
}

# ------------------------
# CROP MODELS
# ------------------------
crop_models = {
    "Rice": {"yield_file": "rice_yield_model.pkl", "prod_file": "rice_prod_model.pkl"},
    "Wheat": {"yield_file": "wheat_yield_model.pkl", "prod_file": "wheat_prod_model.pkl"},
    "Maize": {"yield_file": "maize_yield_model.pkl", "prod_file": "maize_prod_model.pkl"},
    "Cotton": {"yield_file": "cotton_yield_model.pkl", "prod_file": "cotton_prod_model.pkl"},
    "Sugarcane": {"yield_file": "sugarcane_yield_model.pkl", "prod_file": "sugarcane_prod_model.pkl"}
}

# ------------------------
# LOAD YIELD STATS
# ------------------------
try:
    with open("crop_yield_stats.json", "r") as f:
        crop_stats = json.load(f)
except FileNotFoundError:
    crop_stats = {}
    st.warning("⚠️ crop_yield_stats.json not found. Percentage yield will not be available.")

# ------------------------
# SIDEBAR INPUTS
# ------------------------
lang = st.sidebar.selectbox("🌐 " + t("Language", "en"),
                            options=list(language_options.keys()),
                            format_func=lambda x: language_options[x])
st.sidebar.title("🌍 " + t("Farm Settings", lang))

# City & coordinates
if "city" not in st.session_state:
    st.session_state.city = "Hisar"
if "coords" not in st.session_state:
    st.session_state.coords = {"lat": 29.1539, "lon": 75.7229}

city_input = st.sidebar.text_input(t("Enter City:", lang), st.session_state.city)
if city_input != st.session_state.city:
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_input}&limit=1&appid={API_KEY}"
    geo_data = requests.get(geo_url).json()
    if geo_data:
        st.session_state.coords = {"lat": geo_data[0]["lat"], "lon": geo_data[0]["lon"]}
        st.session_state.city = city_input
    else:
        st.sidebar.error("⚠️ " + t("City not found!", lang))

# Crop selection
crop_type = st.sidebar.selectbox(t("Select Crop Type", lang), list(crop_models.keys()))
manual_field_size = st.sidebar.number_input(t("Field Size (hectares)", lang),
                                            min_value=0.1, value=1.0, step=0.1)

st.sidebar.header("🪴 " + t("Soil Parameters", lang))
soil_ph = st.sidebar.slider(t("Soil pH Level", lang), 3.0, 10.0, 6.5, 0.1)
soil_moisture = st.sidebar.selectbox(t("Soil Moisture", lang), ["Low", "Medium", "High"])
soil_fertility = st.sidebar.selectbox(t("Soil Fertility Level", lang), ["Low", "Medium", "High"])

# ------------------------
# IMAGE UPLOAD (NEW)
# ------------------------
st.sidebar.header("📷 " + t("Field Imagery", lang))
uploaded_file = st.sidebar.file_uploader(t("Upload field image (RGB) — optional", lang),
                                         type=["png", "jpg", "jpeg"])
ndvi_preview = None
has_image = 0
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.sidebar.image(image, caption=t("Uploaded Image", lang), use_column_width=True)
        has_image = 1
        # Simple NDVI-like proxy: (G - R) / (G + R) using RGB as approximation (NOT true NDVI)
        arr = np.array(image).astype(float)
        R = arr[..., 0]
        G = arr[..., 1]
        denom = (G + R)
        denom[denom == 0] = 1e-6
        ndvi_proxy = (G - R) / denom
        ndvi_preview = ndvi_proxy.mean()
        st.sidebar.markdown(f"**{t('NDVI proxy (avg):', lang)}** {ndvi_preview:.3f}")
    except Exception as e:
        st.sidebar.error(t("Could not process the image.", lang))

# ------------------------
# TITLE LEFT + LOGO RIGHT
# ------------------------
logo_path = Path(__file__).parent / "logo.png"

col1, col2 = st.columns([6, 3])  # left wide, right narrow

with col1:
    st.markdown(
        """
        <h1 style="text-align:left; font-size: 42px;">🌾 Yield Sense</h1>
        <h3 style="text-align:left; font-size: 24px;">🤖 AI-Powered Crop Yield Prediction</h3>
        """,
        unsafe_allow_html=True
    )

with col2:
    if logo_path.exists():
        st.image(str(logo_path), width=220)

st.markdown(
    t("Get real-time weather data, soil inputs, field imagery and crop-specific recommendations for your farm. ", lang)
)

# ------------------------
# FIELD SELECTION MAP
# ------------------------
st.subheader("🗺️ " + t("Draw Your Field on the Map", lang))
m = folium.Map(location=[st.session_state.coords["lat"], st.session_state.coords["lon"]],
               zoom_start=16, tiles="Esri.WorldImagery")
Draw(draw_options={'polyline': False, 'circle': False,
                   'marker': False, 'circlemarker': False}).add_to(m)
map_data = st_folium(m, width=700, height=500)

field_size_map = None
if map_data and map_data.get("all_drawings"):
    last_drawn = map_data["all_drawings"][-1]
    if last_drawn["type"] in ["polygon", "rectangle"]:
        coords = [(lat, lon) for lon, lat in last_drawn["geometry"]["coordinates"][0]]
        polygon = Polygon(coords)
        # Approx area in hectares (rough)
        field_size_map = polygon.area / 10000
        st.session_state.selected_field_coords = coords
        st.success(t("✅ Field selected successfully!", lang))
        st.write(f"📐 {t('Calculated field size:', lang)} {field_size_map:.2f} ha")

# ------------------------
# WEATHER API
# ------------------------
url_current = f"http://api.openweathermap.org/data/2.5/weather?lat={st.session_state.coords['lat']}&lon={st.session_state.coords['lon']}&units=metric&appid={API_KEY}"
current_data = requests.get(url_current).json()

avg_temp, avg_humidity, total_rainfall = 25, 70, 5  # fallback

if "main" in current_data:
    today_date = datetime.now().strftime("%A, %d %B %Y")
    st.subheader(f"🌦️ {t('Current Weather in', lang)} {st.session_state.city} ({today_date})")
    col1, col2, col3 = st.columns(3)
    temp = current_data['main']['temp']
    humidity = current_data['main']['humidity']
    rain = current_data.get("rain", {}).get("1h", 0)

    avg_temp, avg_humidity, total_rainfall = temp, humidity, rain

    col1.metric("🌡️ " + t("Temperature", lang), f"{temp}°C")
    col2.metric("💧 " + t("Humidity", lang), f"{humidity}%")
    col3.metric("🌧️ " + t("Rainfall (last hr)", lang), f"{rain} mm")
else:
    st.error("⚠️ " + t("Could not fetch current weather.", lang))

# ------------------------
# 5-DAY WEATHER FORECAST
# ------------------------
url_forecast = f"http://api.openweathermap.org/data/2.5/forecast?lat={st.session_state.coords['lat']}&lon={st.session_state.coords['lon']}&units=metric&appid={API_KEY}"
forecast_data = requests.get(url_forecast).json()

if "list" in forecast_data:
    st.subheader("📅 " + t("5-Day Forecast", lang))

    df_forecast = pd.DataFrame(forecast_data["list"])
    df_forecast["dt"] = pd.to_datetime(df_forecast["dt"], unit="s")
    df_forecast["date"] = df_forecast["dt"].dt.date

    daily = df_forecast.groupby("date").agg({
        "main": lambda x: {
            "min": min(i["temp_min"] for i in x),
            "max": max(i["temp_max"] for i in x)
        },
        "weather": lambda x: x.iloc[0][0]["description"],
        "pop": "mean"
    }).reset_index()

    cols = st.columns(len(daily))
    for i, row in daily.iterrows():
        with cols[i]:
            st.markdown(f"**{row['date'].strftime('%a, %d %b')}**")
            st.write(f"🌡️ {row['main']['max']}° / {row['main']['min']}°C")
            st.caption(t(row["weather"].title(), lang))
            st.write(f"🌧️ {row['pop']*100:.0f}% rain chance")
else:
    st.error("⚠️ " + t("Could not fetch 5-day forecast.", lang))

# ------------------------
# YIELD PREDICTION
# ------------------------
last_year_yield = st.sidebar.number_input(
    t("Last Year Yield Production (kg)", lang), 
    min_value=0.0, value=0.0, step=100.0
)

# Recommendation model paths
RECOMM_MODEL_PATH = "recommendation_xgb_model.pkl"
RECOMM_LE_PATH = "rec_label_encoder.pkl"

# utility to encode categorical features
def encode_soil_moisture(x):
    return {"Low": 0, "Medium": 1, "High": 2}.get(x, 1)

def encode_soil_fertility(x):
    return {"Low": 0, "Medium": 1, "High": 2}.get(x, 1)

crop_encoding = {"Rice": 0, "Wheat": 1, "Maize": 2, "Cotton": 3, "Sugarcane": 4}

# Recommendation logic
def get_recommendation(feature_vector):
    # Try to use XGBoost model if available
    if XGB_AVAILABLE:
        try:
            model = joblib.load(RECOMM_MODEL_PATH)
            # if label encoder exists, decode predicted label
            if Path(RECOMM_LE_PATH).exists():
                le = joblib.load(RECOMM_LE_PATH)
                pred = model.predict(np.array(feature_vector).reshape(1, -1))
                label = le.inverse_transform(pred.astype(int))[0]
                return label
            else:
                pred = model.predict(np.array(feature_vector).reshape(1, -1))
                return f"Recommendation code: {int(pred[0])} (no label encoder found)"
        except Exception as e:
            # fallback to rule-based
            pass

    # Rule-based fallback
    ph = feature_vector[0]
    moisture = feature_vector[1]
    fertility = feature_vector[2]
    crop = feature_vector[3]
    ndvi = feature_vector[4]

    recs = []
    if ph < 5.5:
        recs.append("Apply lime to raise pH")
    elif ph > 7.5:
        recs.append("Apply sulfur to lower pH")

    if moisture == 0:
        recs.append("Irrigate: low soil moisture detected")
    elif moisture == 2:
        recs.append("Drain excess water or avoid waterlogging")

    if fertility == 0:
        recs.append("Apply balanced NPK fertilizer")
    elif fertility == 2:
        recs.append("Maintain organic matter, monitor nutrient levels")

    if ndvi is not None:
        if ndvi < 0.05:
            recs.append("Low vegetation vigor — inspect for pests/disease")
        elif ndvi > 0.25:
            recs.append("Good vigor — maintain current management")

    # crop specific hint
    if crop == 0:  # Rice
        recs.append("Rice: ensure proper transplanting depth and puddling management")
    elif crop == 1:
        recs.append("Wheat: consider timely N top dressing at tillering")

    return "; ".join(recs)

# Predict button
if st.button("🔮 " + t("Predict Crop Yield", lang)):
    crop_info = crop_models.get(crop_type)
    if crop_info:
        try:
            # Load models
            yield_model = joblib.load(crop_info["yield_file"])
            prod_model = joblib.load(crop_info["prod_file"])

            # Use map area if drawn, else manual input
            field_size = field_size_map if field_size_map else manual_field_size

            # Map features for yield model (same as original)
            if crop_type == "Rice":
                input_yield = pd.DataFrame({
                    "RICE AREA (1000 ha)": [field_size],
                    "RICE PRODUCTION (1000 tons)": [0]
                })
                input_prod = pd.DataFrame({
                    "RICE AREA (1000 ha)": [field_size],
                    "RICE YIELD (Kg per ha)": [0]
                })
            elif crop_type == "Wheat":
                input_yield = pd.DataFrame({
                    "WHEAT AREA (1000 ha)": [field_size],
                    "WHEAT PRODUCTION (1000 tons)": [0]
                })
                input_prod = pd.DataFrame({
                    "WHEAT AREA (1000 ha)": [field_size],
                    "WHEAT YIELD (Kg per ha)": [0]
                })
            elif crop_type == "Maize":
                input_yield = pd.DataFrame({
                    "MAIZE AREA (1000 ha)": [field_size],
                    "MAIZE PRODUCTION (1000 tons)": [0]
                })
                input_prod = pd.DataFrame({
                    "MAIZE AREA (1000 ha)": [field_size],
                    "MAIZE YIELD (Kg per ha)": [0]
                })
            elif crop_type == "Cotton":
                input_yield = pd.DataFrame({
                    "COTTON AREA (1000 ha)": [field_size],
                    "COTTON PRODUCTION (1000 tons)": [0]
                })
                input_prod = pd.DataFrame({
                    "COTTON AREA (1000 ha)": [field_size],
                    "COTTON YIELD (Kg per ha)": [0]
                })
            elif crop_type == "Sugarcane":
                input_yield = pd.DataFrame({
                    "SUGARCANE AREA (1000 ha)": [field_size],
                    "SUGARCANE PRODUCTION (1000 tons)": [0]
                })
                input_prod = pd.DataFrame({
                    "SUGARCANE AREA (1000 ha)": [field_size],
                    "SUGARCANE YIELD (Kg per ha)": [0]
                })

            # Predict yield
            yield_per_ha = yield_model.predict(input_yield)[0]
            total_yield = yield_per_ha * field_size

            # Update production model input
            input_prod.iloc[0, -1] = yield_per_ha
            predicted_production = prod_model.predict(input_prod)[0]

            # Show results
            st.success(f"{t('Predicted Yield per ha:', lang)} {yield_per_ha:.2f} kg/ha")
            st.success(f"{t('Predicted Total Yield:', lang)} {total_yield:.2f} kg")
            st.success(f"{t('Predicted Production:', lang)} {predicted_production:.2f} tons")

            # Compare with last year
            if last_year_yield > 0:
                diff_percent = ((total_yield - last_year_yield) / last_year_yield) * 100
                if diff_percent > 0:
                    st.success(f"📈 {t('Increase from last year:', lang)} +{diff_percent:.2f}%")
                elif diff_percent < 0:
                    st.error(f"📉 {t('Decrease from last year:', lang)} {diff_percent:.2f}%")
                else:
                    st.info(t("No change compared to last year.", lang))

            # ------------------------
            # RECOMMENDATION (XGBoost)
            # ------------------------
            st.subheader("🔧 " + t("Crop Recommendations", lang))

            feature_vector = [
                soil_ph,
                encode_soil_moisture(soil_moisture),
                encode_soil_fertility(soil_fertility),
                crop_encoding.get(crop_type, 0),
                ndvi_preview if ndvi_preview is not None else -1,
                avg_temp,
                avg_humidity,
                has_image
            ]

            recommendation = get_recommendation(feature_vector)
            st.info(recommendation)

        except FileNotFoundError:
            st.error("⚠️ Model files not found for this crop.")
    else:
        st.error("⚠️ No model available for this crop.")

# ------------------------
# FOOTER
# ------------------------
st.markdown("<hr><p style='text-align:center; color:gray;'>Powered by OpenWeatherMap & Streamlit</p>",
            unsafe_allow_html=True)

# ------------------------
# NOTES FOR USER
# ------------------------
# - To enable the XGBoost-based recommendation system, place 'recommendation_xgb_model.pkl' and optionally
#   'rec_label_encoder.pkl' (if you trained with categorical labels) in the same folder as this app.
# - The recommendation model should accept the feature vector: [soil_ph, soil_moisture_encoded, soil_fertility_encoded,
#   crop_encoded, ndvi_proxy_mean, avg_temp, avg_humidity, has_image]
# - If XGBoost or model files are absent, the app falls back to a simple rule-based recommender.
# - Install dependencies: pip install streamlit folium streamlit-folium deep-translator shapely joblib xgboost pillow
