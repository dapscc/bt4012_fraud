import pandas as pd
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.one_hot import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def get_processed_data():
    ## Global data preprocessing
    df = pd.read_csv("carclaims.csv")
    df = df.drop(columns=['PolicyNumber',"PolicyType"])
    df['Age'] =df['Age'].replace({0:16.5})
    df = df[df["MonthClaimed"]!='0']

    ## Encoding ordinal features
    col_ordering = [{'col':'Month','mapping':{'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}},
        {'col':'DayOfWeek','mapping':{'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}},
        {'col':'DayOfWeekClaimed','mapping':{'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}},
        {'col':'MonthClaimed','mapping':{'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}},
        {'col':'PastNumberOfClaims','mapping':{'none':0 ,'1':1,'2 to 4':2,'more than 4':5 }},
        {'col':'NumberOfSuppliments','mapping':{'none':0,'1 to 2':1,'3 to 5':3,'more than 5':6}}, 
        {'col':'VehiclePrice','mapping':{'less than 20,000':1,'20,000 to 29,000':2,'30,000 to 39,000':3,
                                        '40,000 to 59,000':4,'60,000 to 69,000':5, 'more than 69,000':6}},
        {'col':'AgeOfVehicle','mapping':{'3 years':3,'6 years':6,'7 years':7,'more than 7':8,'5 years':5,'new':0,'4 years':4,'2 years':2}},
        {'col':'Days:Policy-Accident','mapping':{'more than 30':4,'15 to 30':3,'none':0,'1 to 7':1,'8 to 15':2}},
        {'col':'Days:Policy-Claim','mapping':{'more than 30':4,'15 to 30':3,'none':0,'1 to 7':1,'8 to 15':2}},
        {'col':'AgeOfPolicyHolder','mapping':{'16 to 17':1,'18 to 20':2,'21 to 25':3,'26 to 30':4,'31 to 35':5,'36 to 40':6,
                                            '41 to 50':7,'51 to 65':8,'over 65':9}},
        {'col':'AddressChange-Claim','mapping':{'no change':0,'under 6 months':1,'1 year':2,'2 to 3 years':3,'4 to 8 years':4}},
        {'col':'NumberOfCars','mapping':{'1 vehicle':1,'2 vehicles':2,'3 to 4':3,'5 to 8':4,'more than 8':5}}]

    ord_encoder = OrdinalEncoder(mapping = col_ordering, return_df=True)
    df2 = df.copy()
    df2 = ord_encoder.fit_transform(df2)

    ## Encoding nominal features
    onehot = OneHotEncoder(cols=['Make', 'MaritalStatus', 'VehicleCategory', 'BasePolicy'], use_cat_names=True, return_df=True) 
    df3 = onehot.fit_transform(df2)

    df4 = df3.copy()
    df4[['PoliceReportFiled', 'WitnessPresent']] = df3[['PoliceReportFiled', 'WitnessPresent']].replace({'No': 0, 'Yes': 1})
    df4[['AccidentArea']] = df4[['AccidentArea']].replace( {
        'Rural' : 0,
        'Urban' : 1
        })
    df4[['Fault']] = df4[['Fault']].replace( {
        'Third Party' : 0,
        'Policy Holder' : 1
        })
    df4[['Sex']] = df4[['Sex']].replace( {
        'Female' : 0,
        'Male' : 1
        })
    df4[['AgentType']] = df4[['AgentType']].replace({
        'Internal' : 0,
        'External' : 1
        })
    df4[['FraudFound']] = df4[['FraudFound']].replace({
        'No' : 0,
        'Yes' : 1
        })


    df4.to_csv('processed_data.csv', index=False)

    X = df4.drop('FraudFound', axis=1)  # Features
    y = df4['FraudFound']  # Target variable

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print("Training set shape:", X_train.shape, y_train.shape)
    print("Validation set shape:", X_val.shape, y_val.shape)
    print("Test set shape:", X_test.shape, y_test.shape)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return df4, X_train, y_train, X_val, y_val, X_test, y_test