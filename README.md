# Time Series Covid Analysis

## Data

The data used in this analysis is from the Johns Hopkins University CSSE. The data can be found [here](https://github.com/CSSEGISandData/COVID-19). There are a series of jsons containing lat-long information used for creating our maps that can be found in our data folder, [./data/geo_data/](https://github.com/dkeogh1/time-series-covid/data/geo_data). On Macrh 22nd, JHU CSSE switched the format of their data from tracking United States cases at higher level of granularity than on the country level. It previous had state-evel information, and sometimes even county or city-level data as well. For the purposes of building models at that level, there is a also a snaphot csv from their March 22nd release that can be found within the data folder here [./data/jhu_data/](https://github.com/dkeogh1/time-series-covid/data/jhu_data/time_series_19-covid-Confirmed_3_22.csv). For most recent data, I would suggest their most recent confirmed cases data set.

Another data set that the underlying tools in this repo could be repurposed for is the time-series panel-data that The New York Times has released, [here](https://github.com/nytimes/covid-19-data). This data is similaryly formatted to the JHU CSSE data; however it is generally US-centric and makes a point preserving county and state-level information.

## Results

Two sets of models and predictions were made for this analysis. The first set of models trained a LSTM neural network on panel data for country-level cumulative confirmed cases. We then visually analyzed our results folium maps. Below are the cumulative cases world-map, where the the more red the country is labeleled, the more confirmed cases there are. The first displayed is the first day in the data set, January 22nd 2020, the second is the last day of the data set when we trained these models, March 22nd 2020, and the third is the aggregation of model predictions 10 days out from the last day (in this case, April 1st 2020). Above each map is a link ot an html file, which is is an interactive version of the map you're viewing.

### Cumulative Confirmed Cases - World

#### January 22nd - World Confirmed Cases

[Interactive Version](./outputs/map_html/first_day_confirmed_cases.html)

_PNG_:

<img src="./outputs/map_images/first_day_world.png" width="600"/> 

#### March 22nd - World Confirmed Cases

[Interactive Version](./outputs/map_html/last_day_confirmed_cases.html)

_PNG_:

<img src="./outputs/map_images/last_day_world.png" width="600"/> 

#### April 1st - World Confirmed Cases (10 Day-Out Prediction)

_PNG_:

[Interactive Version](./outputs/map_html/10_day_world.html)

<img src="./outputs/map_images/10_day_future_world.png" width="600"/> 

### New Daily Cases - United States

The second set of results is a similar analysis, performed on US states. Instead of analyzing and predicting cumulative cases, though, we create a lagging difference between teh cumulative cases from day to day to infer new-daily cases. Using this, we train models for each state to infer the number of new cases based on each one's respective past daily confirmed cases.

#### January 22nd - US States Daily Cases

[Interactive Version](./outputs/map_html/first_day_new_cases_states.html)

_PNG_:

<img src="./outputs/map_images/first_day_states.png" width="600"/> 

#### March 22nd - US States Daily Cases

[Interactive Version](./outputs/map_html/last_day_new_cases_states.html)

_PNG_:

<img src="./outputs/map_images/last_day_states.png" width="600"/> 

#### April 1st - US States Daily Cases (10 Day-Out Prediction)

_PNG_:

[Interactive Version](./outputs/map_html/10_day_new_cases_states.html)

<img src="./outputs/map_images/10_day_future_states.png" width="600"/> 

