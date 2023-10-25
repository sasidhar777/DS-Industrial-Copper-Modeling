import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import folium
from streamlit_folium import folium_static
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
import math 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



st.title("Guvi's Industrial Copper Modeling")
image = Image.open('guvi_image.jpg')
st.image(image, caption='Guvi Brand Amabassador')



def replace_starting_with_zeros(value):
    if value.startswith('00000'):
        return np.nan
    else:
        return value


def regression():
    st.write('You have choosen for slaes price prediction ')
    #options=['PL','S','W','WI','Others']
    #item_type = st.selectbox('Select one options:', options)
    st.write("Select the item_type:")
    PL = st.checkbox("PL")
    S = st.checkbox("S")
    W = st.checkbox("W")
    WI = st.checkbox("WI")
    Others = st.checkbox("Others")

    item_date = st.text_input("Item date in yyyymmdd format")
    quantity = st.text_input("quantitiy is tones")
    country = st.text_input('country code')
    application = st.text_input('application')
    thickness = st.text_input('thickness')
    width = st.text_input('width')
    delivery_date = st.text_input("delivery date in yyyymmdd format")
    if st.button("Click here for predicting sales price"):
        regression_model_building(PL,S,W,WI,Others , float(item_date) , float(quantity), float(country), float(application), 
        float(thickness) , float(width), float(delivery_date))

def regression_model_building(PL,S,W,WI,Others,item_date,quantity,country,application,thickness,width,delivery_date):
    st.write("Prediction for the given inputs is started ....")
    excel_file_path = 'DataSet\Copper_Set.csv'
    # List of encodings to try
    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']

    # Try reading the Excel file with different encodings
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(excel_file_path, encoding=encoding)
            # If reading succeeds, break out of the loop
            break
        except Exception as e:
            print(f"Failed with encoding '{encoding}': {e}")
    df['material_ref'] = df['material_ref'].astype(str)
    df['material_ref'] = df['material_ref'].apply(replace_starting_with_zeros)
    column_name = 'quantity tons'
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce', downcast='float')
    column_names = ['country', 'item_date', 'customer', 'status' , 'application' ,'quantity tons' , 'thickness' , 'delivery date' , 'selling_price' ]
    # replacing country col null values with median of the column
    for column_name in column_names: 
        df[column_name] = df[column_name].fillna(df[column_name].mode().iloc[0])
    df['material_ref'] = df['material_ref'].bfill() # in case 1st col is null it is covered
    df['material_ref'] = df['material_ref'].ffill()
    columns_to_drop = ['id', 'material_ref','product_ref','customer']
    df = df.drop(columns_to_drop, axis=1)
    dff = pd.get_dummies(df["item type"])
    dff = dff.astype('float')
    dff[['item_date','quantity tons','country','status','application','thickness','width','delivery date',
    'selling_price']] = df[['item_date','quantity tons','country','status','application','thickness','width',
                            'delivery date','selling_price']]
    df_reg = dff.drop('status', axis=1)
    model = LinearRegression()
    X=df_reg.drop('selling_price',axis=1)
    y=df_reg['selling_price']
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    y_train_pred=lr.predict(x_train)
    y_test_pred=lr.predict(x_test)
    #st.write(x_test)
    #st.write(" Root of Train error" , math.sqrt(mean_squared_error(y_train,y_train_pred)))
    #st.write("Root of Test error",math.sqrt(mean_squared_error(y_test,y_test_pred)))

    #prediction of value starts frome here
    item_type = {
        "Others" : int(Others),
        "PL" : int(PL),
        "S" : int(S),
        "W" : int(W),
        "WI" : int(WI)
    }
    features = np.array( list(item_type.values()) + [float(item_date), float(quantity), float(country), float(application), float(thickness), float(width), float(delivery_date)]).reshape(1, -1)  
    # Make a prediction
    prediction = lr.predict(features)
    predicted_selling_price = abs(prediction[0])
    st.write("selling price for the given inputs is: ", predicted_selling_price )


def classification():
    st.write('You have choosen for prediction of status')
    #options=['PL','S','W','WI','Others']
    #item_type = st.selectbox('Select one options:', options)
    st.write("Select the item_type:")
    PL = st.checkbox("PL")
    S = st.checkbox("S")
    W = st.checkbox("W")
    WI = st.checkbox("WI")
    Others = st.checkbox("Others")
    item_date = st.text_input("Item date in yyyymmdd format")
    quantity = st.text_input("quantitiy is tones")
    country = st.text_input('country code')
    application = st.text_input('application')
    thickness = st.text_input('thickness')
    width = st.text_input('width')
    delivery_date = st.text_input("delivery date in yyyymmdd format")
    if st.button("Click here for prediction"):
        #classification_model_building(width)
        classification_model_building(PL,S,W,WI,Others , float(item_date) , float(quantity), float(country), float(application), 
        float(thickness) , float(width), float(delivery_date))



def classification_model_building(PL,S,W,WI,Others,item_date,quantity,country,application,thickness,width,delivery_date):
   
    st.write("Prediction for the given inputs is started ..." )
    excel_file_path = 'DataSet\Copper_Set.csv'
    # List of encodings to try
    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']

    # Try reading the Excel file with different encodings
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(excel_file_path, encoding=encoding)
            # If reading succeeds, break out of the loop
            break
        except Exception as e:
            print(f"Failed with encoding '{encoding}': {e}")
    df['material_ref'] = df['material_ref'].astype(str)
    df['material_ref'] = df['material_ref'].apply(replace_starting_with_zeros)
    column_name = 'quantity tons'
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce', downcast='float')
    column_names = ['country', 'item_date', 'customer', 'status' , 'application' ,'quantity tons' , 'thickness' , 'delivery date' , 'selling_price' ]
    # replacing country col null values with median of the column
    for column_name in column_names: 
        df[column_name] = df[column_name].fillna(df[column_name].mode().iloc[0])
    df['material_ref'] = df['material_ref'].bfill() # in case 1st col is null it is covered
    df['material_ref'] = df['material_ref'].ffill()
    columns_to_drop = ['id', 'material_ref','product_ref','customer']
    df = df.drop(columns_to_drop, axis=1)
    dff = pd.get_dummies(df["item type"])
    dff = dff.astype('float')
    dff[['item_date','quantity tons','country','status','application','thickness','width','delivery date',
    'selling_price']] = df[['item_date','quantity tons','country','status','application','thickness','width',
                            'delivery date','selling_price']]
    df_class = dff.drop('selling_price', axis=1)
    ordinal_cols=['status']
    le=LabelEncoder()
    for col in ordinal_cols:
        le.fit(df_class[col])
        df_class[col]=le.transform(df[col])
    X = df_class.drop('status',axis=1)
    y = df_class["status"]
    # Get the mapping of labels to original values
    label_mapping = dict(zip(le.transform(le.classes_), le.classes_))
    # Convert the mapping to a DataFrame
    mapping_df = pd.DataFrame(label_mapping.items(), columns=["Label", "Original_Value"])

    # Print the DataFrame
    mapping_df=mapping_df.set_index("Label")
    st.write(mapping_df)
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    item_type = {
        "Others" : int(Others),
        "PL" : int(PL),
        "S" : int(S),
        "W" : int(W),
        "WI" : int(WI)
    }
    features = np.array( list(item_type.values()) + [float(item_date), float(quantity), float(country), float(application), float(thickness), float(width), float(delivery_date)]).reshape(1, -1)  
    rf_accuracy, rf_conf_matrix, rf_class_report,predicted_status  = evaluate_model(rf_classifier, X_test, y_test,features)
    #st.write("Random Forest Model:")
    st.write("Accuracy:", rf_accuracy)
    st.write("The predicted status is: " , predicted_status , " and it's value is " , mapping_df.iloc[predicted_status,0] )
    


def evaluate_model(model, X_test, y_test,features):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    prediction = model.predict(features)
    predicted_status = prediction[0]
    return accuracy, conf_matrix, class_report, predicted_status



if __name__ == "__main__":
    options=[ 'Select one option','classification For Status prediction','Regression for selling price prediction']
    selected_option = st.selectbox('Select one option:', options)
    if selected_option== 'Regression for selling price prediction':
        regression()
    elif selected_option== 'classification For Status prediction' :
        classification()