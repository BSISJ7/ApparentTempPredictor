import json
import os
from os.path import exists
import numpy as numpyFuncs
import pandas as pandasFuncs
import plotly
import plotly.express as px
import plotly.graph_objs as go
import requests as apiRequests
from flask import Flask, render_template, request
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from math import sqrt
import sklearn.linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import joblib




def setupCityInfo():
    csvFile = os.getcwd() + '/templates/assets/TempData/World Cities Lat Lon (Small).csv'
    cityData = pandasFuncs.read_csv(csvFile)
    cityDict = {}
    cityNamesList = []
    for x in range(0, len(cityData), 5):
        cityDict[cityData['city'][x]] = [cityData['country'][x],cityData['lat'][x],cityData['lng'][x]]
        cityNamesList.append(cityData['city'][x])
    cityNamesList.sort()

    return cityNamesList, cityDict



global cityNamesList
global cityDict
cityNamesList, cityDict = setupCityInfo() 


global selectedCity
selectedCity = cityNamesList[0]


global barGraph
global scatterFig
global parallelGraph


global TEMPERATURE_DATA_CSV
TEMPERATURE_DATA_CSV = os.getcwd()+'/templates/assets/TempData/Weather in Szeged 2006-2016 (copy).csv'


def trainModel():
    trainingData = pandasFuncs.read_csv(TEMPERATURE_DATA_CSV)
    origData = trainingData
    trainingData = trainingData.dropna()
    trainingData = trainingData.drop_duplicates()
    trainingData.drop(trainingData[trainingData['Humidity'] == 0].index, inplace=True)   


# ========================================== Adjust Pressure Data ========================================== #

    # trainingData.loc[trainingData['Pressure (millibars)'] == 0, 'Pressure (millibars)'] = .01
    Q1 = trainingData["Pressure (millibars)"].quantile(0.10)
    Q3 = trainingData["Pressure (millibars)"].quantile(0.95)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    trainingData['Pressure (millibars)'] = \
        numpyFuncs.where(trainingData['Pressure (millibars)'] > upper_limit, upper_limit,
                                         trainingData['Pressure (millibars)'])
    trainingData['Pressure (millibars)'] = \
        numpyFuncs.where(trainingData['Pressure (millibars)'] < lower_limit, lower_limit,
                                         trainingData['Pressure (millibars)'])


# ========================================== Drop And Concat Summary Precip Data ========================================== #

    trainingData = trainingData.drop(['Daily Summary', 'Loud Cover', 'Formatted Date'], axis=1)
    del trainingData['Summary']
    del trainingData['Precip Type']

# ========================================== Split Main And Apparent Temp Data ========================================== #

    tempColumn = pandasFuncs.DataFrame(trainingData['Apparent Temperature (C)'])
    trainingData = trainingData.drop(['Apparent Temperature (C)'], axis='columns')
    mainTrainingData, mainTestData, appTempTrainData, appTempTestData = \
        train_test_split(trainingData, tempColumn, test_size=0.2)


# ========================================== Wind Spd, Humidity, Visibility Transformation ========================================== #

    sqrt_transformer = FunctionTransformer(numpyFuncs.sqrt)
    transformedWindSpd = sqrt_transformer.transform(trainingData['Wind Speed (km/h)'])
    trainingData['Wind Speed (km/h)'] = transformedWindSpd

    expTransformer = FunctionTransformer(numpyFuncs.exp)
    transformedHumidity = expTransformer.transform(trainingData['Humidity'])
    trainingData['Humidity'] = transformedHumidity

    expTransformer = FunctionTransformer(numpyFuncs.exp)
    transformedVisibility = expTransformer.transform(trainingData['Visibility (km)'])
    trainingData['Visibility (km)'] = transformedVisibility


# ========================================== Wind Bearing Data Scaling ========================================== #
    discretizer = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')
    windTrainingData = pandasFuncs.DataFrame(trainingData, columns=['Wind Bearing (degrees)'])
    discretizer.fit(windTrainingData)
    discWindTrainData = pandasFuncs.DataFrame(discretizer.transform(windTrainingData))
    trainingData['Wind Bearing (degrees)'] = discWindTrainData


# ========================================== Data Standardization ========================================== #

    dataColumnNames = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)',
                          'Visibility (km)', 'Pressure (millibars)']
    standardization = StandardScaler()
    standardization.fit(trainingData[dataColumnNames])
    transformedDataScaled = standardization.transform(trainingData[dataColumnNames])
    trainingData[dataColumnNames] = transformedDataScaled


# ========================================== Principle Component Analysis ========================================== #

    principleCompAnalysis = PCA(n_components=6)
    principleCompAnalysis.fit(mainTrainingData)
    mainTrainPCA = principleCompAnalysis.transform(mainTrainingData)
    principleCompAnalysis.fit(mainTestData)
    mainTestPCA = principleCompAnalysis.transform(mainTestData)
    pcaDataFrame = pandasFuncs.DataFrame(mainTrainPCA)

# ========================================== Principle Component Analysis ========================================== #
    linModel = LinearRegression()
    model = linModel.fit(mainTrainPCA, appTempTrainData)
    print("Model Score:",model.score(mainTestPCA, appTempTestData))

    predictions = linModel.predict(mainTestPCA)
    y_hat = pandasFuncs.DataFrame(predictions, columns=["predicted"])
    print("Mean Squared Error:",mean_squared_error(appTempTestData, y_hat))

    rmsq = sqrt(mean_squared_error(appTempTestData, y_hat))
    print("Mean Squared Error Sqrt:",rmsq)

    print("Linear Coeficient:",linModel.coef_)
    
# ========================================== Create Linear Regression Model ========================================== #
    
    linModel = LinearRegression()
    linModel.fit(mainTrainPCA, appTempTrainData)
    joblib_file = "joblib_RL_Model.pkl"  
    joblib.dump(linModel, joblib_file)






joblib_file = "joblib_RL_Model.pkl"
if not exists(joblib_file):
    trainModel()
    








def prepareWeatherData(latitude, longitude):
# ========================================== Collect Weather Data ========================================== #


    API_KEY = '0b5c51277ca90405ac816ebeb8d4b9c6'
    BASE_URL = "https://api.openweathermap.org/data/2.5/onecall?lat="+latitude+"&lon="+longitude+"&exclude=daily,minutely,current&" \
    "units=metric&appid="+API_KEY



    weatherData = apiRequests.get(BASE_URL).json()
    fullWeatherDict = {'Temperature (C)':[], 'Humidity':[], 'Wind Speed (km/h)':[], 'Wind Bearing (degrees)':[],
           'Visibility (km)':[], 'Pressure (millibars)':[]}

    for x in range(0, len(weatherData['hourly'])):
        fullWeatherDict['Temperature (C)'].append(weatherData['hourly'][x]['temp'])
        fullWeatherDict['Humidity'].append(weatherData['hourly'][x]['humidity'])
        fullWeatherDict['Wind Speed (km/h)'].append(weatherData['hourly'][x]['wind_speed'])
        fullWeatherDict['Wind Bearing (degrees)'].append(weatherData['hourly'][x]['wind_deg'])
        fullWeatherDict['Visibility (km)'].append(weatherData['hourly'][x]['visibility']/1000)
        fullWeatherDict['Pressure (millibars)'].append(weatherData['hourly'][x]['pressure'])

    panDataFrame = pandasFuncs.DataFrame(fullWeatherDict)


    # ========================================== Modify Pressure ========================================== #


    Q1 = panDataFrame["Pressure (millibars)"].quantile(0.10)
    Q3 = panDataFrame["Pressure (millibars)"].quantile(0.95)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    panDataFrame['Pressure (millibars)'] = \
    numpyFuncs.where(panDataFrame['Pressure (millibars)'] > \
                     upper_limit, upper_limit, panDataFrame['Pressure (millibars)'])
    panDataFrame['Pressure (millibars)'] = \
    numpyFuncs.where(panDataFrame['Pressure (millibars)'] < \
                     lower_limit, lower_limit, panDataFrame['Pressure (millibars)'])


    # ========================================== Transform Weather Data ========================================== #


    sqrt_transformer = FunctionTransformer(numpyFuncs.sqrt)
    transformedWindSpd = sqrt_transformer.transform(panDataFrame['Wind Speed (km/h)'])
    panDataFrame['Wind Speed (km/h)'] = transformedWindSpd

    expTransformer = FunctionTransformer(numpyFuncs.exp)
    transformedHumidity = expTransformer.transform(panDataFrame['Humidity'])
    panDataFrame['Humidity'] = transformedHumidity

    expTransformer = FunctionTransformer(numpyFuncs.exp)
    transformedVisibility = expTransformer.transform(panDataFrame['Visibility (km)'])
    panDataFrame['Visibility (km)'] = transformedVisibility


    # ========================================== Wind Bearing Data Scaling ========================================== #


    discretizer = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')
    windTrainingData = pandasFuncs.DataFrame(panDataFrame, columns=['Wind Bearing (degrees)'])
    discretizer.fit(windTrainingData)
    discWindTrainData = pandasFuncs.DataFrame(discretizer.transform(windTrainingData))
    panDataFrame['Wind Bearing (degrees)'] = discWindTrainData


    # ========================================== Data Standardization ========================================== #


    dataColumnNames = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)',
                      'Visibility (km)', 'Pressure (millibars)']
    standardization = StandardScaler()
    standardization.fit(panDataFrame[dataColumnNames])
    transformedDataScaled = standardization.transform(panDataFrame[dataColumnNames])
    panDataFrame[dataColumnNames] = transformedDataScaled


    # ========================================== Load Model And Predict Outcome ========================================== #

    joblib_file = "joblib_RL_Model.pkl"  
    joblib_LR_model = joblib.load(joblib_file)
    prediction = joblib_LR_model.predict(panDataFrame)
    predictionList = []
    for x in range(0, len(prediction)):predictionList.append(prediction[x][0])


    # ========================================== Get Temperature And Date ========================================== #


    tempDateDict = {'Date':[], 'Temperature':[]}
    for x in range(0, len(weatherData['hourly'])):
        date = datetime.fromtimestamp(float(weatherData['hourly'][x]['dt']))
        tempDateDict['Date'].append(date.strftime("%m-%d %H:00"))
        tempDateDict['Temperature'].append(weatherData['hourly'][x]['temp'])

    return fullWeatherDict, tempDateDict, predictionList


    # ========================================== Setup Data Displays ========================================== #


ASSETS_FOLDER = 'templates/assets'
app = Flask(__name__, static_folder=ASSETS_FOLDER)
#app.debug = True



@app.route('/bar')
def getBarGraph(dataDict, predictionList):
    global barGraph

    trace1 = go.Bar(
       x = dataDict['Date'],
       y = dataDict['Temperature'],
       name = 'Temperature'
    )
    trace2 = go.Bar(
       x = dataDict['Date'],
       y = predictionList,
       name = 'Apparent Temperature'
    )
    data = [trace1, trace2]
    layout = go.Layout(barmode = 'group')
    fig = go.Figure(data = data, layout = layout)
    barGraph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return barGraph


@app.route('/scatter')
def getScatterGraph(fullWeatherDict):
    global scatterFig
    df = pandasFuncs.DataFrame(fullWeatherDict)
    scatterFig = px.scatter(df, x="Wind Speed (km/h)", y="Wind Bearing (degrees)", color="Temperature (C)",
                 size='Temperature (C)', hover_data=['Humidity'])
    scatterPlot = json.dumps(scatterFig, cls=plotly.utils.PlotlyJSONEncoder)
    return scatterPlot


@app.route('/parallel')
def getParallelGraph(fullWeatherDict, predictionList):
    global parallelGraph
    for x in range(0, len(predictionList)):
        predictionList[x] = int(predictionList[x])
    df = px.data.tips()
    fullPredictDict = fullWeatherDict
    fullPredictDict['Predictions'] = predictionList
    fig = px.parallel_categories(fullPredictDict, color="Predictions", color_continuous_scale=px.colors.sequential.Inferno,width=1500, height=800)
    parallelGraph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return parallelGraph


@app.route('/', methods=['GET'])
def index():
    global selectedCity
    global cityDict
    global cityNamesList

    if request.args.get('jsdata') is not None:
        selectedCity = request.args.get('jsdata')

    fullWeatherDict, tempDateDict, predictionList = prepareWeatherData(str(cityDict[selectedCity][0]), str(cityDict[selectedCity][1]))

    return render_template('index.html', cityList=cityNamesList, scatterGraph=getScatterGraph(fullWeatherDict),
        parallelGraph=getParallelGraph(fullWeatherDict, predictionList), barGraph=getBarGraph(tempDateDict, predictionList))
