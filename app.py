# # #streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import joblib
from datetime import date, timedelta

st.set_page_config(
    page_title="Weekly Sales Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# تحميل الملف من Google Drive تلقائيًا
FILE_ID = "1NGQ7EzZ2RAiU8_Sbyh3vZRF4qepwzgDR"
FILE_NAME = "rf_model_compressed2.pkl"

def download_model():
    if not os.path.exists(FILE_NAME):
        st.info("Downloading model file...")
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        response = requests.get(url)
        with open(FILE_NAME, "wb") as f:
            f.write(response.content)
        st.success("Model downloaded successfully!")

download_model()

@st.cache_resource
def load_model():
    model, columns = joblib.load(FILE_NAME)
    return model, columns

model, model_columns = load_model()

def get_weeks_in_month(year: int, month: int):
    first = date(year, month, 1)
    last = date(year + 1, 1, 1) - timedelta(days=1) if month == 12 else date(year, month + 1, 1) - timedelta(days=1)
    weeks = set()
    curr = first
    while curr <= last:
        weeks.add(curr.isocalendar()[1])
        curr += timedelta(days=1)
    return sorted(weeks)

with st.sidebar:
    st.header("User Inputs")
    store        = st.number_input("Store", min_value=1, value=1)
    dept         = st.number_input("Dept", min_value=1, value=1)
    type_map     = {'A':1, 'B':2, 'C':3}
    type_val     = st.selectbox("Type", ['A', 'B', 'C'])
    size         = st.number_input("Size", min_value=0, value=100000)
    cpi          = st.number_input("CPI", value=220.0)
    unemployment = st.number_input("Unemployment", value=7.0)
    temperature  = st.number_input("Temperature", value=60.0)
    fuel_price   = st.number_input("Fuel_Price", value=3.0)
    markdown1    = st.number_input("MarkDown1", value=0.0)
    markdown2    = st.number_input("MarkDown2", value=0.0)
    markdown3    = st.number_input("MarkDown3", value=0.0)
    markdown4    = st.number_input("MarkDown4", value=0.0)
    markdown5    = st.number_input("MarkDown5", value=0.0)
    is_holiday   = st.selectbox("IsHoliday", ["No", "Yes"])
    year         = st.number_input("Year", min_value=2010, max_value=2030, value=date.today().year)
    month        = st.number_input("Month", min_value=1, max_value=12, value=date.today().month)
    weeks        = get_weeks_in_month(year, month)
    week         = st.selectbox("Week", weeks)

holidays = {
    'Super_Bowl':   1 if week == 5  else 0,
    'Labor_Day':    1 if week == 36 else 0,
    'Thanksgiving': 1 if week == 47 else 0,
    'BlackFriday':  1 if week == 48 else 0,
    'Christmas':    1 if week == 52 else 0,
}

bool_map = {'Yes': 1, 'No': 0}
input_dict = {
    'Store': store,
    'Dept': dept,
    'Type': type_map[type_val],
    'Size': size,
    'CPI': cpi,
    'Unemployment': unemployment,
    'Temperature': temperature,
    'Fuel_Price': fuel_price,
    'MarkDown1': markdown1,
    'MarkDown2': markdown2,
    'MarkDown3': markdown3,
    'MarkDown4': markdown4,
    'MarkDown5': markdown5,
    'IsHoliday': bool_map[is_holiday],
    'year': year,
    'month': month,
    'week': week,
    'Super_Bowl': holidays['Super_Bowl'],
    'Labor_Day': holidays['Labor_Day'],
    'Thanksgiving': holidays['Thanksgiving'],
    'BlackFriday': holidays['BlackFriday'],
    'Christmas': holidays['Christmas'],
}

final_input = {col: input_dict.get(col, 0) for col in model_columns}
input_df = pd.DataFrame([final_input], columns=model_columns).astype(float)

st.header("Prediction Result")
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    st.metric("Weekly Sales (Predicted)", f"{pred:,.2f}")




