from flask import Flask, request, render_template
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import pickle
import pandas as pd


data=pd.read_csv('clean_quikr_data.csv')

companies=sorted(data['company'].unique())
car_models=sorted(data['name'].unique())
kms_driven=sorted(data['kms_driven'].unique())
year=sorted(data['year'].unique())
fuel_type=data['fuel_type'].unique()

ohe=OneHotEncoder(sparse=False,drop='first')
ohe.fit(data[['name','company','fuel_type']])
print(ohe.categories_)

cm=make_column_transformer((OneHotEncoder(categories=ohe.categories_,drop='first',sparse=False),['name','company','fuel_type']),remainder='passthrough')

with open('Car_Price_Predictor','rb') as f:
        car_predictor = pickle.load(f)

app=Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    model=request.args.get('model')
    company=request.args.get('company')
    kms=request.args.get('kms')
    fuel=request.args.get('fuel')
    car_year=request.args.get('year')
    # print(model,company,kms,fuel,car_year)
    if model is None:
      return render_template('index.html',companies=companies,car_models=car_models,car_years=year,fuel_type=fuel_type,kms_drive=kms_driven)
    elif company is None:
         return render_template('index.html',companies=companies,car_models=car_models,car_years=year,fuel_type=fuel_type,kms_drive=kms_driven)
    elif kms is None:
      return render_template('index.html',companies=companies,car_models=car_models,car_years=year,fuel_type=fuel_type,kms_drive=kms_driven)
    elif fuel is None:
      return render_template('index.html',companies=companies,car_models=car_models,car_years=year,fuel_type=fuel_type,kms_drive=kms_driven)
    elif car_year is None:
      return render_template('index.html',companies=companies,car_models=car_models,car_years=year,fuel_type=fuel_type,kms_drive=kms_driven)
    else:
        predictions=car_predictor.predict(cm.fit_transform(pd.DataFrame([[model,company,car_year,kms,fuel]],columns=['name','company','year','kms_driven','fuel_type'])))
        predictions=round(predictions[0])
        return render_template('index.html',companies=companies,car_models=car_models,car_years=year,fuel_type=fuel_type,kms_drive=kms_driven, pre=predictions)


if __name__=="__main__":
   app.run(debug=True)

