from flask import Flask,request,jsonify
import pandas as pd
import joblib
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

model = joblib.load("./model.pkl")
scaler = joblib.load("./scaler.pkl")
NAME = "FLAG_MOBIL,FLAG_CONT_MOBILE,FLAG_EMP_PHONE,REGION_RATING_CLIENT_W_CITY,REGION_RATING_CLIENT,CNT_FAM_MEMBERS,FLAG_OWN_CAR,FLAG_PHONE,AMT_REQ_CREDIT_BUREAU_YEAR,REG_CITY_NOT_WORK_CITY,FLAG_WORK_PHONE,DAYS_EMPLOYED,EMPLOYMENT_YEARS,LIVE_CITY_NOT_WORK_CITY,CNT_CHILDREN,AMT_INCOME_TOTAL,OBS_30_CNT_SOCIAL_CIRCLE,OBS_60_CNT_SOCIAL_CIRCLE,AMT_GOODS_PRICE,AMT_REQ_CREDIT_BUREAU_QRT,FLAG_EMAIL"

def predict(data):
    name=NAME+"\n"+",".join(data)
    sio = StringIO(name)
    dfs = pd.read_csv(sio)
    df = pd.DataFrame(scaler.transform(dfs),columns=dfs.columns,index=dfs.index)
    prediction = model.predict(df)
    return bool(prediction[0])
    
# predict("1,1,1,3,3,1.0,1,1,1.0,0,0,107,0.29315068493150687,0,0,202500.0,5.0,5.0,450000.0,0.0,0".split(","))
app = Flask(__name__)
@app.route("/predict", methods=['GET'])
def predictor():
    data = request.args.get('params','')
    prediction = predict(data.split(" "))
    return jsonify(prediction)

def main():
    app.run(host = "0.0.0.0",port = 3000,debug = True)

if(__name__=='__main__'):
    main()
