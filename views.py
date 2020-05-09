from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.conf import settings
from django.db.models import Q
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.db import connection
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import itertools

# Create your views here.

def details(request, fileId):
    ### Get the File name ###
    fileDetails = getFileData(fileId)
    
    #### Read CSV File and Upload into Database ####
    csv_data = pd.read_excel(str(os.getcwd())+'/media/'+ fileDetails['files_original_file_name'])
    csv_data_list = csv_data.values.tolist()

    context = {
        "saleslist": csv_data_list,
        "headers": list(csv_data.columns),
        "collenth": len(list(csv_data.columns)),
        "salesitems": list(csv_data.Category.unique()) 
    }

    # Message according medicines Role #
    context['heading'] = "Sales Details"
    return render(request, 'sales-list.html', context)

def category(request, fileId):
    ### Get the File name ###
    fileDetails = getFileData(fileId)
    
    #### Read CSV File and Upload into Database ####
    csv_data = pd.read_excel(str(os.getcwd())+'/media/'+ fileDetails['files_original_file_name'])

    context = {
        "salesitems": list(csv_data.Category.unique()),
        "files_id":  fileId
    }

    # Message according medicines Role #
    context['heading'] = "Sales Category Details"
    return render(request, 'category-list.html', context)

def prediction(request, fileId, category):
    ### Get the File name ###
    fileDetails = getFileData(fileId)
    
    #### Read CSV File and Upload into Database ####
    csv_data = pd.read_excel(str(os.getcwd())+'/media/'+ fileDetails['files_original_file_name'])
    
    # # Get Historical Data
    csv_data = csv_data.loc[csv_data['Category'] == category]
    # csv_data['Order Date'].min(), csv_data['Order Date'].max()
    cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
    csv_data.drop(cols, axis=1, inplace=True)
    
    csv_data = csv_data.sort_values('Order Date')
    csv_data = csv_data.groupby('Order Date')['Sales'].sum().reset_index()
    csv_data = csv_data.set_index('Order Date')
    csv_data.index
    csv_data = csv_data['Sales'].resample('MS').mean()
    
      
    current_data = csv_data.values
    current_label = csv_data.index
    
    
    # ## Arima Integration ##
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
            
    mod = sm.tsa.statespace.SARIMAX(csv_data,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    
    results = mod.fit()

    # Prediction data testing for 12 months
    prediction_data = results.get_prediction(start=pd.to_datetime('2016-01-01'), dynamic=False)
    
    # Forcast next 12 months data
    forecasting = (results.get_forecast(steps=12)).predicted_mean
    print(type(forecasting))
    context = {   
        "forcating_data": forecasting.to_dict(),
        "historical_data" : csv_data.to_dict(),
        "category": category,
        "current_data": cumSumToString(current_data),
        "current_label": current_label.astype('str').tolist(),
        "forecasting_label": current_label.astype('str').tolist()+((forecasting.index).astype('str').tolist()),
        "forecasting_data":  cumSumToString(current_data) + cumSumToString(forecasting),
    }

    # Message according medicines Role #
    context['heading'] = "Sales Details"
    return render(request, 'prediction.html', context)
    
def listToString(list):
    lst=[]
    for i in list:
        lst.append(i[0])
    return lst

def dateToString(list):
    lst=[]
    for i in list:
        ts = pd.to_datetime(str(i[0]))
        d = ts.strftime('%d-%m-%Y')
        lst.append(d)
    return lst

def cumSumToString(list):
    lst=[]
    for i in list:
        lst.append(i)
    return lst
    
def dictfetchall(cursor):
    "Return all rows from a cursor as a dict"
    columns = [col[0] for col in cursor.description]
    return [
        dict(zip(columns, row))
        for row in cursor.fetchall()
    ]

def getFileData(id):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM files WHERE files_id = " + id)
    dataList = dictfetchall(cursor)
    return dataList[0]

def dataTraining(df):
    train_data, test_data = df[0:int(len(df)*0.8)], df[int(len(df)*0.8):]
    train_ar = train_data['Close'].values
    test_ar = test_data['Close'].values
    history = [x for x in train_ar]
    predictions = list()
    for t in range(len(test_ar)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_ar[t]
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test_ar, predictions)
    print('Testing Mean Squared Error: %.3f' % error)
    error2 = smape_kun(test_ar, predictions)
    print('Symmetric mean absolute percentage error: %.3f' % error2)
    return predictions



def smape_kun(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))))