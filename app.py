import streamlit as st
import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Ignore warnings
warnings.filterwarnings('ignore')

# CSS to add background image
background_image_css = """
<style>
    .stApp {
        background-image: url('https://static.vecteezy.com/system/resources/previews/027/004/037/non_2x/green-natural-leaves-background-free-photo.jpg');  /* Replace with your image URL */
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
</style>
"""

# Inject the CSS into the Streamlit app
st.markdown(background_image_css, unsafe_allow_html=True)

# Streamlit app title and description
st.title("AGRI-FORECAST")
st.header("Predict Future Values of Agri-Horticulture Commodities")
st.subheader("Enter the Zone, Season, Commodity, Month, and Year for which you need predictions, and get the predicted average sales for that month.")

# Zone selection box
zone = st.selectbox(
    "ZONE",
    ["North", "South", "East", "West", "Central", "Northeast"]
)

# Season selection box
season = st.selectbox(
    "SEASON",
    ["Winter", "Spring", "Summer", "Monsoon", "Autumn", "Pre-Winter"]
)

# Commodity selection box
com = st.selectbox(
    "COMMODITY", 
    ["Gram Dal", "Sugar", "Gur", "Wheat", "Tea", "Milk", "Salt", "Atta", "Tur/Arhar Dal", "Urad Dal", "Moong Dal", 
     "Masoor Dal", "Groundnut Oil", "Mustard Oil", "Vanaspati", 
     "Sunflower Oil", "Soya Oil", "Palm Oil","Rice", "Potato", 
     "Onion", "Tomato"]
)

# Load data and train SARIMA model functions (cached)
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['Month', 'Sales']
    df.dropna(inplace=True)
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    return df

@st.cache_resource
def train_sarima_model(df, order, seasonal_order):
    model = SARIMAX(df['Sales'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    return model_fit

# SARIMA Model Configuration for Different Commodities
file_path = None  # Initialize file_path
if com == "Gram Dal":
    file_path = "Dal_Price.csv"
    order = (0, 1, 2)
    seasonal_order = (0, 1, 2, 12)  # Seasonality assumed on a yearly basis (12 months)
elif com == "Sugar":
    file_path = "Chini.csv"
    order = (0, 1, 1)
    seasonal_order = (0, 1, 1, 12)
elif com == "Wheat":
    file_path = "Wheat - Sheet1.csv"
    order = (1, 1, 0)
    seasonal_order = (1, 1, 0, 12)
elif com == "Gur":
    file_path = "Gur.csv"
    order = (2, 1, 1)
    seasonal_order = (2, 1, 1, 12)
elif com == "Milk":
    file_path = "Milk - Sheet1.csv"
    order = (2, 2, 2)
    seasonal_order = (2, 2, 2, 12)
elif com == "Tea":
    file_path = "Tea - Sheet1.csv"
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
elif com == "Salt":
    file_path = "Salt - Sheet1.csv"
    order = (0, 1, 1)
    seasonal_order = (0, 1, 1, 12)
else:
    st.write("Model is in progress.....")

# Proceed only if the file_path is defined (i.e., for the commodities that are ready)
if file_path:
    # Common Functionality for Forecasting and Displaying Results
    try:
        # Load the data
        df = load_data(file_path)

        # Train the SARIMA model
        model_fit = train_sarima_model(df, order, seasonal_order)

        # Forecasting for 12 months (1 year ahead)
        forecast12 = model_fit.get_forecast(steps=120)
        forecast_values_12 = forecast12.predicted_mean

        # Create a date range for the forecasted values
        forecast_dates12 = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=120, freq='MS')
        forecast_values_12.index = forecast_dates12

        # Combine historical and forecast data
        combined_series = pd.concat([df['Sales'], forecast_values_12])

        # Allow the user to input past or future dates
        st.subheader('Get Prediction or Historical Value for a Specific Month and Year')

        # Allow user to select any year and month within the range of historical and forecast data
        min_year = combined_series.index.min().year
        max_year = combined_series.index.max().year
        year = st.number_input("Enter the Year", min_value=min_year, max_value=max_year, value=min_year)
        month = st.selectbox("Enter the Month", list(range(1, 13)), index=0)

        # Add a button to get the prediction or historical value
        if st.button("Get Value"):
            # Construct the date string for lookup
            date = f"{year}-{month:02d}-01"

            # Show the value for the selected date
            try:
                value = combined_series.loc[date]
                if pd.Timestamp(date) in df.index:
                    st.subheader(f"Historical Sales for {date}: {value:.2f}")
                else:
                    st.subheader(f"Predicted Sales for {date}: {value:.2f}")
            except KeyError:
                st.error("The selected date is out of the available range.")

    except FileNotFoundError:
        st.error("The specified file was not found. Please check the file path and try again.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
