import streamlit as st
import pandas as pd
import numpy as np
import sys
import time
import altair as alt
import datetime
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go

# Interactive Streamlit elements, like these sliders, return their value.
# This gives you an extremely simple interaction model.
#import plotly.graph_objects as go
today = datetime.datetime.today()
yesterday = datetime.date.today()+ datetime.timedelta(days=-1)
TODAY_DATE = str(today.year) + '-' + str(today.month) + '-' + str(today.day)
YESTERDAY_DATE = str(yesterday.year) + '-' + str(yesterday.month) + '-' + str(yesterday.day)
_SUSCEPTIBLE_COLOR = "rgba(130,130,230,.4)"
_RECOVERED_COLOR = "rgba(280,100,180,.4)"
TEMPLATE = "plotly_white"
COLOR_MAP = {
    "default": "#262730",
    "pink": "#E22A5B",
    "purple": "#985FFF",
    "susceptible": _SUSCEPTIBLE_COLOR,
    "recovered": _RECOVERED_COLOR,
}

def _set_legends(fig):
    fig.layout.update(legend=dict(x=-0.1, y=1.2))
    fig.layout.update(legend_orientation="h")


def generate_html(
    text,
    color=COLOR_MAP["default"],
    bold=False,
    font_family=None,
    font_size=None,
    line_height=None,
    tag="div",
):
    if bold:
        text = f"<strong>{text}</strong>"
    css_style = f"color:{color};"
    if font_family:
        css_style += f"font-family:{font_family};"
    if font_size:
        css_style += f"font-size:{font_size};"
    if line_height:
        css_style += f"line-height:{line_height};"

    return f"<{tag} style={css_style}>{text}</{tag}>"



st.title('Go corona! {}'.format(sys.version))

def plot_historical_data(df):
    # Convert wide to long

    df = pd.melt(
        df,	
        id_vars="date",
        value_vars=["CumConfirmed", "CumDeaths", "CumRecovered"],
        var_name="Status",
        value_name="Number",
    )

    fig = px.scatter(
        df, x="date", y="Number", color="Status", template=TEMPLATE, opacity=0.8
    )

    _set_legends(fig)

    return fig


DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

baseURL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
def loadData(fileName, columnName): 
    data = pd.read_csv(baseURL + fileName) \
             .drop(['Lat', 'Long'], axis=1) \
             .melt(id_vars=['Province/State', 'Country/Region'], var_name='date', value_name=columnName) \
             .fillna('<all>')
    data['date'] = data['date'].astype('datetime64[ns]')
    return data

allData = loadData("time_series_19-covid-Confirmed.csv", "CumConfirmed") \
    .merge(loadData("time_series_19-covid-Deaths.csv", "CumDeaths")) \
    .merge(loadData("time_series_19-covid-Recovered.csv", "CumRecovered"))
COUNTRIES = list(set(allData['Country/Region']))
SELECTED_COUNTRIES = None
allData.to_csv('AllData.csv')
# https://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz
# https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv
# https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv
# https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv
@st.cache
def load_data():
    # data = pd.read_csv(DATA_URL, nrows=nrows)
    # lowercase = lambda x: str(x).lower()
    # data.rename(lowercase, axis='columns', inplace=True)
    # data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    daily_report = pd.read_csv('COVID-19-master/csse_covid_19_data/csse_covid_19_daily_reports/03-21-2020.csv')
    # time_series_confirmed = pd.read_csv('COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
    # time_series_recovered = pd.read_csv('COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
    # time_series_deaths = pd.read_csv('COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')
    return daily_report
data_load_state = st.text('Loading data...')
daily_report = load_data()
COUNTRIES = list(set(daily_report['Country/Region']))
data_load_state.text('Loading data... done!')

# map_data = data.rename(columns={'Latitude':'lat','Longitude':'lon','Last Update':'date/time','Country/Region':'base'})
# map_data = map_data[['date/time','lat','lon','base']]

# ts_map_data = time_series.rename(columns={'Lat':'lat','Long':'lon','Country/Region':'base'})
# ts_map_data = ts_map_data[['lat','lon','base']]
# st.map(ts_map_data)
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(daily_report)

#st.map(filtered_data)   
SELECTED_COUNTRIES = st.sidebar.selectbox(
    'Select your country?',
     options=COUNTRIES)
ret = daily_report[daily_report['Country/Region'] == SELECTED_COUNTRIES].sum()
date_last_fetched = datetime.datetime.now()
st.sidebar.markdown(
            body=generate_html(
                text=f"Statistics refreshed as of ",
                line_height=0,
                font_family="Arial",
                font_size="12px",
            ),
            unsafe_allow_html=True,
        )
st.sidebar.markdown(
    body=generate_html(text=f"{date_last_fetched}", bold=True, line_height=0),
    unsafe_allow_html=True,
     )

st.sidebar.markdown(
        body=generate_html(
        text=f'Population: {int(12222222):,}<br>Infected: {int(ret["Confirmed"]):,}<br>'
        f'Recovered: {int(ret["Recovered"]):,}<br>Dead: {int(ret["Deaths"]):,}',
        line_height=0,
        font_family="Arial",
        font_size="0.9rem",
        tag="p",
        ),
        unsafe_allow_html=True,
        )

st.subheader(f"How has the disease spread in {SELECTED_COUNTRIES}?")
fig = plot_historical_data(allData[allData['Country/Region'] == SELECTED_COUNTRIES])
st.write(fig)

# st.subheader('Raw Time Series Data')
# st.write(allData)

