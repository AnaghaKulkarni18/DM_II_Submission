import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
import sklearn
import xgboost
from numpy import asarray

app = Flask(__name__)

# data = pd.read_csv('CleanedData.csv')

model = joblib.load('HousePricePred_model.pkl')

# model = pickle.load(open('HP_Trained_Model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    LotArea = int(request.form['LotArea'])
    GrLivArea = int(request.form['GrLivArea'])
    
    FstFlrSF = int(request.form['FstFlrSF'])
    GarageArea = int(request.form['GarageArea'])
    
    BsmtUnfSF = int(request.form['BsmtUnfSF'])
    TotalBsmtSF = int(request.form['TotalBsmtSF'])
    
    LotFrontage = float(request.form['LotFrontage'])
    YearBuilt = int(request.form['YearBuilt'])
    
    MoSold = int(request.form['MoSold'])
    GarageYrBlt = float(request.form['GarageYrBlt'])
    
    BsmtFinSF1 = int(request.form['BsmtFinSF1'])
    YearRemodAdd = int(request.form['YearRemodAdd'])
    
    OpenPorchSF = int(request.form['OpenPorchSF'])
    WoodDeckSF = int(request.form['WoodDeckSF'])
    
    
    Neighborhood = request.form['Neighborhood']
    if (Neighborhood == 'CollgCr'):
    	Neighborhood = 0
    elif (Neighborhood == 'Veenker'):
    	Neighborhood = 1
    elif (Neighborhood == 'Crawfor'):
    	Neighborhood = 2
    elif (Neighborhood == 'NoRidge'):
    	Neighborhood = 3
    elif (Neighborhood == 'Mitchel'):
    	Neighborhood = 4
    elif (Neighborhood == 'Somerst'):
    	Neighborhood = 5
    elif (Neighborhood == 'NWAmes'):
    	Neighborhood = 6       
    elif (Neighborhood == 'OldTown'):
    	Neighborhood = 7
    elif (Neighborhood == 'BrkSide'):
    	Neighborhood = 8 
    elif (Neighborhood == 'Sawyer'):
    	Neighborhood = 9
    elif (Neighborhood == 'NridgHt'):
    	Neighborhood = 10
    elif (Neighborhood == 'NAmes'):
    	Neighborhood = 11
    elif (Neighborhood == 'SawyerW'):
    	Neighborhood = 12
    elif (Neighborhood == 'IDOTRR'):
    	Neighborhood = 13
    elif (Neighborhood == 'MeadowV'):
    	Neighborhood = 14       
    elif (Neighborhood == 'Edwards'):
    	Neighborhood = 15
    elif (Neighborhood == 'Timber'):
    	Neighborhood = 16    
    elif (Neighborhood == 'Gilbert'):
    	Neighborhood = 17
    elif (Neighborhood == 'StoneBr'):
    	Neighborhood = 18
    elif (Neighborhood == 'ClearCr'):
    	Neighborhood = 19
    elif (Neighborhood == 'NPkVill'):
    	Neighborhood = 20
    elif (Neighborhood == 'Blmngtn'):
    	Neighborhood = 21
    elif (Neighborhood == 'BrDale'):
    	Neighborhood = 22       
    elif (Neighborhood == 'SWISU'):
    	Neighborhood = 23
    elif (Neighborhood == 'Blueste'):
    	Neighborhood = 25  
      
    MasVnrArea = float(request.form['MasVnrArea'])
    OverallQual = int(request.form['OverallQual'])
    ScndFlrSF = int(request.form['ScndFlrSF'])
    TotRmsAbvGrd = int(request.form['TotRmsAbvGrd'])
    
  
    Exterior1st = request.form['Exterior1st']
    if (Exterior1st == 'VinylSd'):
    	Exterior1st = 0
    elif (Exterior1st == 'HdBoard'):
    	Exterior1st = 1
    elif (Exterior1st == 'MetalSd'):
    	Exterior1st = 2
    elif (Exterior1st == 'Wd Sdng'):
    	Exterior1st = 3
    elif (Exterior1st == 'Plywood'):
    	Exterior1st = 4
    elif (Exterior1st == 'CemntBd'):
    	Exterior1st = 5
    elif (Exterior1st == 'BrkFace'):
    	Exterior1st = 6       
    elif (Exterior1st == 'WdShing'):
    	Exterior1st = 7
    elif (Exterior1st == 'Stucco'):
    	Exterior1st = 8 
    elif (Exterior1st == 'AsbShng'):
    	Exterior1st = 9
    elif (Exterior1st == 'BrkComm'):
    	Exterior1st = 10
    elif (Exterior1st == 'Stone'):
    	Exterior1st = 11
    elif (Exterior1st == 'CBlock'):
    	Exterior1st = 12
    elif (Exterior1st == 'ImStucc'):
    	Exterior1st = 13
    elif (Exterior1st == 'AsphShn'):
    	Exterior1st = 14       
        
    feature_list = [LotArea, GrLivArea, FstFlrSF, GarageArea, BsmtUnfSF, TotalBsmtSF, 
                                       LotFrontage, YearBuilt, MoSold, GarageYrBlt,BsmtFinSF1,YearRemodAdd, 
                                       OpenPorchSF, WoodDeckSF, Neighborhood, MasVnrArea, OverallQual, ScndFlrSF, 
                                       TotRmsAbvGrd, Exterior1st]
    
    float_features = [float(x) for x in feature_list]
    final_features = np.array([float_features])
    prediction = model.predict(final_features)

    
    output = round(prediction[0], 5)

    return render_template('index.html', prediction_text='Estimated House Price should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
