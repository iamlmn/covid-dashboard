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
import numpy as np
BED_DATA_PATH = "data/world_bank_bed_data.csv"

# Interactive Streamlit elements, like these sliders, return their value.
# This gives you an extremely simple interaction model.
#import plotly.graph_objects as go
today = datetime.datetime.today()
yesterday = datetime.date.today()+ datetime.timedelta(days=-1)
TODAY_DATE = str(today.year) + '-' + str(today.month) + '-' + str(today.day)
YESTERDAY_DATE = str(yesterday.year) + '-' + str(yesterday.month) + '-' + str(yesterday.day)
DATA_DATE = TODAY_DATE
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

def _get_latest_bed_estimate(row):
    non_empty_estimates = [float(x) for x in row.values if float(x) > 0]
    try:
        return non_empty_estimates[-1]
    except IndexError:
        return np.nan


def preprocess_bed_data(path):
    df = pd.read_csv(path, header=2)
    df.rename({"Country Name": "Country/Region"}, axis=1, inplace=True)
    df.drop(["Country Code", "Indicator Name", "Indicator Code"], axis=1, inplace=True)
    df.set_index("Country/Region", inplace=True)
    # Beds are per 1000 people
    df["Latest Bed Estimate"] = df.apply(_get_latest_bed_estimate, axis=1) / 1000

    # Rename countries to match demographics and disease data
    df = df.rename(
        index={
            "Iran, Islamic Rep.": "Iran",
            "Korea, Rep.": "Korea, South",
            "Russian Federation": "Russia",
            "Egypt, Arab Rep.": "Egypt",
            "Slovak Republic": "Slovakia",
            "Congo, Dem. Rep.": "Congo (Kinshasa)",
            # "Brunei Darussalam": "Brunei",
        }
    )

    return df

def _set_legends(fig):
    fig.layout.update(legend=dict(x=-0.1, y=1.2))
    fig.layout.update(legend_orientation="h")

@st.cache
def peoples_comparison_chart(total_population, Confirmed, Recovered, Deaths):
    """
    A horizontal bar chart comparing # of beds available compared to 
    max number number of beds needed
    """

    df = pd.DataFrame(
        {
            "Label": ["Total Population ", "Confirmed", "Recovered", "Deaths"],
            "Value": [total_population, Confirmed, Recovered, Deaths],
            "Text": [f"{total_population:,}  ", f"{Confirmed:,}  ", f"{Recovered:,}  ", f"{Deaths:,}  "],
            "Color": ["b", "r" ,"g", "y"],
        }
    )
    fig = px.bar(
        df,
        x="Value",
        y="Label",
        color="Color",
        text="Text",
        orientation="h",
        opacity=0.7,
        template=TEMPLATE,
        height=300,
    )

    fig.layout.update(
        showlegend=False,
        xaxis_title="",
        xaxis_showticklabels=False,
        yaxis_title="",
        yaxis_showticklabels=True,
        font=dict(family="Arial", size=15, color=COLOR_MAP["default"]),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)

    return fig



def top_n_column_comparison_bar_chart(df,column_name, color, num_cols = 10):
    """
    A horizontal bar chart comparing # of beds available compared to 
    max number number of beds needed
    """

    df = pd.DataFrame(
        {
            "Label": df['Countries'],
            "Value": df[column_name],
            # "Text": [f"{total_population:,}  ", f"{Confirmed:,}  ", f"{Recovered:,}  ", f"{Deaths:,}  "],
            "Color" : [color for i in range(0,num_cols)],
            "Text": [f"{i:,}" for i in df[column_name]]
        }
    )
    fig = px.bar(
        df,
        x="Value",
        y="Label",
        color="Color",
        text="Text",
        orientation="h",
        opacity=0.5,
        template=TEMPLATE,
        height=500,
    )

    fig.layout.update(
        showlegend=False,
        xaxis_title="",
        xaxis_showticklabels=False,
        yaxis_title="",
        yaxis_showticklabels=True,
        font=dict(family="Arial", size=10, color=COLOR_MAP["default"]),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)

    return fig


def infection_graph(df, y_max):

    # We cannot explicitly set graph width here, have to do it as injected css: see interface.css
    fig = go.Figure(layout=dict(template=TEMPLATE))

    susceptible, infected, recovered = (
        df.loc[df.Status == "Susceptible"],
        df.loc[df.Status == "Infected"],
        df.loc[df.Status == "Recovered"],
    )
    fig.add_scatter(
        x=susceptible.Days,
        y=susceptible.Forecast,
        fillcolor=COLOR_MAP["susceptible"],
        fill="tozeroy",
        mode="lines",
        line=dict(width=0),
        name="Uninfected",
        opacity=0.5,
    )

    fig.add_scatter(
        x=recovered.Days,
        y=recovered.Forecast,
        fillcolor=COLOR_MAP["recovered"],
        fill="tozeroy",
        mode="lines",
        line=dict(width=0),
        name="Recovered",
        opacity=0.5,
    )

    fig.add_scatter(
        x=infected.Days,
        y=infected.Forecast,
        fillcolor="#FFA000",
        fill="tozeroy",
        mode="lines",
        line=dict(width=0),
        name="Infected",
        opacity=0.5,
    )
    fig.update_yaxes(range=[0, y_max])
    fig.layout.update(xaxis_title="Days")
    _set_legends(fig)
    return fig


def reported_cases_count(totalConfirmed, totalDeaths, totalRecovered):
    _border_color = "light-gray"
    _number_format = "font-size:35px; font-style:bold;"
    _cell_style = f" border: 2px solid {_border_color}; border-bottom:2px solid white; margin:10px"
    st.markdown(
        f"<table style='width: 100%; font-size:14px;  border: 0px solid gray; border-spacing: 10px;  border-collapse: collapse;'> "
        f"<tr> "
        f"<td style='{_cell_style}'> Confirmed Cases</td> "
        f"<td style='{_cell_style}'> Deaths </td>"
        f"<td style='{_cell_style}'> Recovered </td>"
        "</tr>"
        f"<tr style='border: 2px solid {_border_color}'> "
        f"<td style='border-right: 2px solid {_border_color}; border-spacing: 10px; {_number_format + 'font-color:red'}' > {totalConfirmed}</td> "
        f"<td style='{_number_format + 'color:red'}'> {int(totalDeaths):,} </td>"
         f"<td style='{_number_format + 'color:green'}'> {int(totalRecovered):,} </td>"
        "</tr>"
        "</table>"
        "<br>",
        unsafe_allow_html=True,
    )

@st.cache
def country_reported_cases_count(totalConfirmed, totalDeaths, totalRecovered):
    _border_color = "light-gray"
    _number_format = "font-size:15px; font-style:bold;"
    _cell_style = f" border: 1px solid {_border_color}; border-bottom:1px solid white; margin:5px"
    st.markdown(
        f"<table style='width: 50%; font-size:7px;  border: 0px solid gray; border-spacing: 5px;  border-collapse: collapse;'> "
        f"<tr> "
        f"<td style='{_cell_style}'> Confirmed Cases in {SELECTED_COUNTRY}</td> "
        f"<td style='{_cell_style}'> Deaths in {SELECTED_COUNTRY}</td>"
        f"<td style='{_cell_style}'> Recovered in {SELECTED_COUNTRY}</td>"
        "</tr>"
        f"<tr style='border: 2px solid {_border_color}'> "
        f"<td style='border-right: 2px solid {_border_color}; border-spacing: 10px; {_number_format + 'font-color:red'}' > {totalConfirmed}</td> "
        f"<td style='{_number_format + 'color:red'}'> {int(totalDeaths):,} </td>"
         f"<td style='{_number_format + 'color:green'}'> {int(totalRecovered):,} </td>"
        "</tr>"
        "</table>"
        "<br>",
        unsafe_allow_html=True,
    )



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



st.title('Corona! {}'.format(sys.version))

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

def plot_mortality_slash_recovery_rate(df):
    # Convert wide to long

    df = pd.melt(
        df, 
        id_vars="Dates",
        value_vars=["RecoveryRate", "MortalityRate"],
        var_name="Status",
        value_name="Percentage",
    )

    fig = px.scatter(
        df, x="Dates", y="Percentage", color="Status", template=TEMPLATE, opacity=0.8
    )

    _set_legends(fig)

    return fig


DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

baseURL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
@st.cache
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
SELECTED_COUNTRY = None
# https://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz
# https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv
# https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv
# https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv
# https://github.com/KaggleDS/covid19-global
@st.cache
def load_time_series_data():
    # data = pd.read_csv(DATA_URL, nrows=nrows)
    # lowercase = lambda x: str(x).lower()
    # data.rename(lowercase, axis='columns', inplace=True)
    # data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    daily_report = pd.read_csv('COVID-19-master/csse_covid_19_data/csse_covid_19_daily_reports/03-21-2020.csv')
    time_series_confirmed = pd.read_csv('COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
    time_series_recovered = pd.read_csv('COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
    time_series_deaths = pd.read_csv('COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')
    return daily_report, time_series_confirmed, time_series_recovered, time_series_deaths
data_load_state = st.text('Loading data...')
daily_report, time_series_confirmed, time_series_recovered, time_series_deaths = load_time_series_data()
COUNTRIES = list(set(daily_report['Country/Region']))
data_load_state.text('Loading data... done!')

# map_data = data.rename(columns={'Latitude':'lat','Longitude':'lon','Last Update':'date/time','Country/Region':'base'})
# map_data = map_data[['date/time','lat','lon','base']]

# ts_map_data = time_series.rename(columns={'Lat':'lat','Long':'lon','Country/Region':'base'})
# ts_map_data = ts_map_data[['lat','lon','base']]
# st.map(ts_map_data)

# Time Series of Coronna data for the selected country
@st.cache
def calclate_country_stats(time_series_confirmed, time_series_recovered, time_series_deaths):
    series_select_countires_confirmed = []
    series_select_countires_recovered = []
    series_select_countires_deaths = []
    for i in COUNTRIES:
        series_select_countires_confirmed.append(time_series_confirmed[time_series_confirmed['Country/Region'] == i]['3/21/20'].sum())
        series_select_countires_recovered.append(time_series_recovered[time_series_recovered['Country/Region'] == i]['3/21/20'].sum())
        series_select_countires_deaths.append(time_series_deaths[time_series_deaths['Country/Region'] == i]['3/21/20'].sum())
    series_mortality_rate = [(x/y)*100 for x, y in zip(series_select_countires_deaths, series_select_countires_confirmed)]
    series_recovery_rate = [(x/y)*100 for x, y in zip(series_select_countires_recovered, series_select_countires_confirmed)]
    df_country_stats = pd.DataFrame(columns = ['Countries','Confirmed','Deaths','Recovered','MortalityRate','RecoveryRate'])
    df_country_stats['Countries'] = pd.Series(COUNTRIES)
    df_country_stats['Confirmed'] = pd.Series(series_select_countires_confirmed)
    df_country_stats['Deaths'] = pd.Series(series_select_countires_deaths)
    df_country_stats['Recovered'] = pd.Series(series_select_countires_recovered)
    df_country_stats['MortalityRate'] = pd.Series(series_mortality_rate)
    df_country_stats['RecoveryRate'] = pd.Series(series_recovery_rate)
    return df_country_stats
#df_country_stats.to_csv('dsadd.csv')
df_country_stats = calclate_country_stats(time_series_confirmed, time_series_recovered, time_series_deaths)
totalConfirmed = df_country_stats['Confirmed'].sum()
totalDeaths = df_country_stats['Deaths'].sum()
totalRecovered = df_country_stats['Recovered'].sum()

# counts
reported_cases_count(totalConfirmed, totalDeaths, totalRecovered)

# map
data = pd.DataFrame({
    'awesome cities' : daily_report['Country/Region'],
    'lat' : daily_report['Latitude'],
    'lon' : daily_report['Longitude']
})

# Adding code so we can have map default to the center of the data
midpoint = (np.average(data['lat']), np.average(data['lon']))

st.deck_gl_chart(
            viewport={
                'latitude': midpoint[0],
                'longitude':  midpoint[1],
                'zoom': 2
            },
            layers=[{
                'type': 'ScatterplotLayer',
                'data': data,
                'radiusScale': 10,
   'radiusMinPixels': 2,
                'getFillColor': [240, 96, 96],
            }]
        )


if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(daily_report)

@st.cache
def time_series_plot_world(allData):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=allData.date, y=allData['CumConfirmed'], name="Infected",
                             line_color='deepskyblue'))

    fig.add_trace(go.Scatter(x=allData.date, y=allData['CumRecovered'], name="Recovered",
                             line_color='green'))

    fig.add_trace(go.Scatter(x=allData.date, y=allData['CumDeaths'], name="Deaths",
                             line_color='red'))

    fig.update_layout(title_text='Time Series of Corona effects total', template=TEMPLATE,  
                      xaxis_rangeslider_visible=True)
    return fig

st.write(time_series_plot_world(allData))



st.subheader('Top 20 countries with highest cases confirmed')
top_20_df = df_country_stats.sort_values(by=['Confirmed'], ascending=False)[:20]
if st.checkbox('Show raw data of top 20 countries with highest cases confirmede'):
    st.subheader('Raw data')
    st.write(top_20_df)
st.write(top_n_column_comparison_bar_chart(top_20_df,'Confirmed','r', 20))

st.subheader('Top 10 countires with highest mortality rate from 20 top country confirmed')
top_10_mortal = top_20_df.sort_values(by=['Deaths'], ascending=False)[:10]
if st.checkbox('Show raw data of top 10 countires with highest mortality rate from 20 top country confirme'):
    st.subheader('Raw data')
    st.write(top_10_mortal)
st.write(top_n_column_comparison_bar_chart(top_10_mortal,'MortalityRate','b',10))




def plot_bar_top_ten_moral(top_10_mortal):
    x=top_10_mortal['Countries']
    fig = go.Figure(go.Bar(x=x, y=top_10_mortal['Confirmed'], name='Confirmed'))
    fig.add_trace(go.Bar(x=x, y=top_10_mortal['Recovered'], name='Recovered'))
    fig.add_trace(go.Bar(x=x, y=top_10_mortal['Deaths'], name='Deaths'))

    fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})
    return fig

st.write(plot_bar_top_ten_moral(top_10_mortal))

#st.map(filtered_data)   
SELECTED_COUNTRY = st.selectbox(
    'Select your country?',
     options=COUNTRIES)
ret = daily_report[daily_report['Country/Region'] == SELECTED_COUNTRY].sum()
date_last_fetched = datetime.datetime.now()
country_reported_cases_count(ret["Confirmed"], ret["Deaths"], ret["Recovered"])
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

Confirmed, Recovered, Deaths = int(ret["Confirmed"]), int(ret["Recovered"]), int(ret["Deaths"])
@st.cache
def calculate_bed_data_and_population():
# Population
    pop_df = pd.read_csv('data/demographics.csv')
    total_population = int(pop_df[pop_df['Country/Region'] == SELECTED_COUNTRY]['Population']) if SELECTED_COUNTRY in list(pop_df['Country/Region']) else 'NA'
    bed_data = preprocess_bed_data(BED_DATA_PATH).reset_index()

    try:
        beds_ratio_in_country = bed_data[bed_data['Country/Region'] == SELECTED_COUNTRY]['Latest Bed Estimate']
    except:
        beds_ratio_in_country = 'NA'

    top_10_pop_df = pop_df.sort_values(by=['Population'], ascending=False)[:10]
    return int(total_population), int(beds_ratio_in_country), top_10_pop_df

total_population, beds_ratio_in_country, top_10_pop_df = calculate_bed_data_and_population()
if str(total_population != 'NA') and str(beds_ratio_in_country != 'NA'):
    beds_in_your_country = beds_ratio_in_country * total_population
    st.write("The total population of {} is {} and the nummber of Physicians availabe are {}. The important variable for hospitals is the  \
        peak number of people who require hospitalization and ventilation at any one time.Your country has around {} beds. Bear in mind that most \
        of these are probably already in use for people sick for other reasons.".format(SELECTED_COUNTRY, total_population, 123, beds_ratio_in_country))
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

st.subheader(f"How has the disease spread in {SELECTED_COUNTRY}?")
fig = plot_historical_data(allData[allData['Country/Region'] == SELECTED_COUNTRY])
st.write(fig)


# st.subheader(f"{SELECTED_COUNTRY}s MoratalityRate & RecoveryRate obsereved")
# fig2 = plot_mortality_slash_recovery_rate(df_country_stats)
# st.write(fig2)

#st.bar_chart(df_recovery)

peoples_comparison_chart = peoples_comparison_chart(total_population, Confirmed, Recovered, Deaths)
st.write(peoples_comparison_chart)


# st.subheader('Raw Time Series Data')
# st.write(allData)



population_string = "Population density is not very high"
if SELECTED_COUNTRY in list(top_10_pop_df['Country/Region']):
    pop_flag = True
    population_string = '{} seems to be in the top 10 highly populated countries.'.format(SELECTED_COUNTRY)

if SELECTED_COUNTRY in list(top_10_mortal['Countries']):
    mortal_flag = True
    mortal_string = 'The mortality rate for your country seems to be {}, which is high.'.format(float(top_10_mortal[top_10_mortal['Countries'] == SELECTED_COUNTRY]['MortalityRate']))
else:
    mortal_string = 'The mortality rate for your country due to nCov seems to be low at the moment.'
    
if SELECTED_COUNTRY in list(top_20_df['Countries']):
    top_cases_flag = True
    if pop_flag and top_cases_flag:
        cases_string = "Social Distancing and vigilant attention to safety is a must for your country. With the dense population the virus could be easily spread."
    else:
        cases_string= "Although its not highly populated, the number of cases reported are high. Please be safe!"
else:
    cases_string= "The condition is not that severe compared to other countries. Please follow social distancing and avoid the spread!"
        
st.write("{}.{}.{}".format(population_string, mortal_string, cases_string))
## Summary
def survey():
    st.write("Try taking a survey to know your severity and possibility of catching the disease")
    # st.checkbox('Fever')

    age = st.sidebar.slider('How old are you?', 0, 130, 25)
    
    gender = st.sidebar.selectbox(
            'Gender?',
            ('normal','Mild Fever', 'High Fever'))
    

    fever = st.sidebar.selectbox(
            'Fever symptoms?',
            ('Male','Female', 'Others'))
    

    condition = st.sidebar.selectbox(
            'How have your symptoms progressed over the last 48 hrs?',
            ('better','improved', 'worsened', 'worsened completely'))
    

    contancts = st.sidebar.multiselect(
            'Travel history or Foreign contacts?',
             ('no travel hiistory', 'No contact with symtoms', "history of travel", "covid patient link"))
    

    diseases = st.sidebar.multiselect(
            'Any known diseases?',
             ('Diabetes', 'High BP', 'Lung disease', 'Stroke' , 'Heart disease', 'Kidney disease', 'Reduced immunity', 'None of these'))
    

    agree = st.sidebar.button('Predict?')
    if agree:
        st.write("I'm ", age, 'years old')
        st.write('You selected:', gender)
        st.write('You selected:', fever)
        st.write('You selected:', condition)
        st.write('You selected:', contancts)
        st.write('You selected:', diseases)

        def get_severity():
            return 'Medium'
        
        st.write('SEVERITY PREDICTION', get_severity())    
if st.sidebar.checkbox('Survey?'):
    st.sidebar.subheader('Please answer these questions')
    survey()