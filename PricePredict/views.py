from django.shortcuts import render
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
from io import BytesIO
import base64


model = load('./savedModels/model.joblib')
data = pd.read_csv('./ML_notebook/cleanx.csv')
data.drop('index', axis=1, inplace=True)

# Create your views here.


def home(request):
    return render(request, 'home.html')


def pricepredict(request):
    if request.method == "POST":
        stops = {'Non-stop': 0, '1': 1, '2': 2, '3': 3, '4': 4}
        date = request.POST["date"]
        Journey_day = int(pd.to_datetime(
                date, format="%Y-%m-%dT%H:%M").day)
        Journey_month = int(pd.to_datetime(
                date, format="%Y-%m-%dT%H:%M").month)
        Journey_hour = int(pd.to_datetime(
                date, format="%Y-%m-%dT%H:%M").hour)
        Journey_minute = int(pd.to_datetime(
                date, format="%Y-%m-%dT%H:%M").minute)
        
        duration = request.POST["duration"]

        duration_datetime = pd.to_datetime(duration, format="%H:%M")
        Duration_hour = int(duration_datetime.hour)
        Duration_min = int(duration_datetime.minute)

        source_name = request.POST['source']
        destination_name = request.POST['destination']

        if source_name == destination_name:
            return render(request, 'predict.html', {'alert_msg': 'Source and Destination cannot be same'})
        elif Duration_hour < 2:
            return render(request, 'predict.html', {'alert_msg': 'Duration should be atleast of 2 Hours'})
        else:
            pred_list = []
            stop_list = []
            airline_col = []
            stop_col = []
            airline_name = request.POST['airline']
            stops_name = request.POST['stops']
            Total_Stops = stops[stops_name]

            airline_index = np.where(data.columns == airline_name)[0][0]
            source_index = np.where(
                data.columns == 'source_' + source_name)[0][0]
            destination_index = np.where(
                data.columns == 'destination_' + destination_name)[0][0]
            X = np.zeros(len(data.columns))
            X[0] = Total_Stops
            X[1] = Journey_day
            X[2] = Journey_month
            X[3] = Journey_hour
            X[4] = Journey_minute
            X[5] = Duration_hour
            X[6] = Duration_min

            if airline_index >= 0 and source_index >= 0 and destination_index >= 0:
                X[airline_index] = 1
                X[source_index] = 1
                X[destination_index] = 1
                y_pred = model.predict([X])[0]

                airline_arr = ['Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Multiple carriers',
                               'Multiple carriers Premium economy', 'SpiceJet', 'Vistara', 'Vistara Premium economy', 'Air Asia']

                airline_arr1 = ['Air\nIndia', 'GoAir', 'IndiGo', 'Jet\nAirways', 'Multiple\ncarriers',
                                'Multiple\ncarriers\nPremium\neconomy', 'SpiceJet', 'Vistara', 'Vistara\nPremium\neconomy', 'Air\nAsia']
                for i in range(len(airline_arr)):
                    if airline_name == airline_arr[i]:
                        pred_list.append(round(y_pred, 2))
                        airline_col.append('crimson')
                    else:
                        air_index = np.where(
                            data.columns == airline_arr[i])[0][0]
                        Y = np.zeros(len(data.columns))
                        Y[0] = Total_Stops
                        Y[1] = Journey_day
                        Y[2] = Journey_month
                        Y[3] = Journey_hour
                        Y[4] = Journey_minute
                        Y[5] = Duration_hour
                        Y[6] = Duration_min
                        Y[air_index] = 1
                        # print(Y)
                        y_prediction = model.predict([Y])[0]
                        pred_list.append(round(y_prediction, 2))
                        airline_col.append('dodgerblue')

                for i in stops:
                    if Total_Stops == stops[i]:
                        stop_list.append(round(y_pred, 2))
                        stop_col.append('crimson')
                    else:

                        Z = np.zeros(len(data.columns))
                        Z[0] = stops[i]
                        Z[1] = Journey_day
                        Z[2] = Journey_month
                        Z[3] = Journey_hour
                        Z[4] = Journey_minute
                        Z[5] = Duration_hour
                        Z[6] = Duration_min
                        Z[airline_index] = 1
                        z_prediction = model.predict([Z])[0]
                        stop_list.append(round(z_prediction, 2))
                        stop_col.append('dodgerblue')

                def create_bar_graph(x, y, xlabel, ylabel, title, colors):
                    plt.clf()
                    plt.figure(figsize=[10, 8])
                    plt.bar(x, y, color=colors)
                    plt.xlabel(xlabel, fontsize=14)
                    plt.ylabel(ylabel, fontsize=14)
                    plt.title(title, fontsize=17)

                    # to add text on top of bar graph
                    for i in range(len(x)):
                        plt.text(i, y[i]+120, y[i], ha='center')


                    buf = BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)

                    # Convert the image to a base64 string
                    image_base64 = base64.b64encode(
                        buf.read()).decode('utf-8').replace("\n", "")

                    return image_base64

                # print(airline_arr, pred_list)
                graph1 = create_bar_graph(
                    airline_arr1, pred_list, 'Airlines', 'Prices', 'Comparison of Airline Prices', airline_col)

                graph2 = create_bar_graph(
                    list(stops.keys()), stop_list, 'Stops', 'Prices', 'Comparison of Ticket Price based on Number of Stops', stop_col)

                return render(request, 'predict.html', {'predicted_value': round(y_pred, 2), 'graph1': graph1, 'graph2': graph2})
                # return render(request, 'predict.html', {'predicted_value': round(y_pred, 2)})

    return render(request, 'predict.html', {'alert_msg': 'there is no data to predict'})
