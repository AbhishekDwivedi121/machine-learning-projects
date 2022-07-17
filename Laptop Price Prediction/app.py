from operator import mod
from flask import Flask,request,render_template
import pandas as pd
import pickle



data = pd.read_csv('new_data.csv')

company = sorted(data['Company'].unique())
inches = sorted(data['Inches'])
Ram = sorted(data['Ram'].unique())
Gpu = sorted(data['Gpu'].unique())
display = sorted(data['display'].unique())
processor = sorted(data['processor'].unique())
HDD = sorted(data['HDD'].unique())
SSD = sorted(data['SSD'].unique())


with open('model.pkl','rb') as f:
    model = pickle.load(f)





app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def index():
    Company = request.args.get('company')
    inche = request.args.get('inches')
    ram = request.args.get('ram')
    gpu = request.args.get('gpu')
    Processor = request.args.get('processor')
    Display = request.args.get('display')
    ssd = request.args.get('ssd')
    hdd = request.args.get('hdd')

    if Company is None:
        return render_template('index.html',hdd=HDD,company=company,inch=inches,ram=Ram,gpu=Gpu,display=display,processor=processor,ssd=SSD)
    elif inche is None:
         return render_template('index.html',hdd=HDD,company=company,inch=inches,ram=Ram,gpu=Gpu,display=display,processor=processor,ssd=SSD)
    elif ram is None:
         return render_template('index.html',hdd=HDD,company=company,inch=inches,ram=Ram,gpu=Gpu,display=display,processor=processor,ssd=SSD)
    elif gpu is None:
         return render_template('index.html',hdd=HDD,company=company,inch=inches,ram=Ram,gpu=Gpu,display=display,processor=processor,ssd=SSD)
    elif Display is None:
         return render_template('index.html',hdd=HDD,company=company,inch=inches,ram=Ram,gpu=Gpu,display=display,processor=processor,ssd=SSD)
    elif ssd is None:
         return render_template('index.html',hdd=HDD,company=company,inch=inches,ram=Ram,gpu=Gpu,display=display,processor=processor,ssd=SSD)
    elif hdd is None:
         return render_template('index.html',hdd=HDD,company=company,inch=inches,ram=Ram,gpu=Gpu,display=display,processor=processor,ssd=SSD)
    else:
        predictions = model.predict(pd.DataFrame([[Company,inche,ram,gpu,Display,Processor,hdd,ssd]],columns=['Company','Inches','Ram','Gpu','display','processor','HDD','SSD']))
        predictions = predictions[0]
        return render_template('index.html',hdd=HDD,company=company,inch=inches,ram=Ram,gpu=Gpu,display=display,processor=processor,ssd=SSD,pre=predictions)
    


   







if __name__ =='__main__':
    app.run(debug=True)