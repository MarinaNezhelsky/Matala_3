import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pickle
from car_data_prep import prepare_data
from sklearn.metrics import mean_squared_error
import os


app = Flask(__name__)

# Define RMSE Scoring Function
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Load the trained model
with open("trained_model.pkl", 'rb') as f:
    car_model = pickle.load(f)

Train_set = pd.read_csv("Train_set.csv")

def encoding(df, reference_df):
    # Calculate the mean price for each category in the reference data
    model_mean_prices = reference_df.groupby('model')['Price'].mean()
    manufactor_mean_prices = reference_df.groupby('manufactor')['Price'].mean()
    overall_mean_price = reference_df['Price'].mean()  # Mean price of all data
    
    def get_model_encoded(row):
        model_price = model_mean_prices.get(row['model'], np.nan)
        if pd.isna(model_price):
            manufactor_price = manufactor_mean_prices.get(row['manufactor'], np.nan)
            return manufactor_price
        return model_price

    df['manufactor_encoded'] = df['manufactor'].map(manufactor_mean_prices)
    df['manufactor_encoded'].fillna(overall_mean_price, inplace=True)
    df['car_model_encoded'] = df.apply(get_model_encoded, axis=1)
    df['gear_encoded'] = df['Gear'].map(reference_df.groupby('Gear')['Price'].mean())
    df['Engine_type_encoded'] = df['Engine_type'].map(reference_df.groupby('Engine_type')['Price'].mean())
    df['prev_encoded'] = df['Prev_ownership'].map(reference_df.groupby('Prev_ownership')['Price'].mean())
    df['prev_encoded'].fillna(overall_mean_price, inplace=True)
    df['curr_encoded'] = df['Curr_ownership'].map(reference_df.groupby('Curr_ownership')['Price'].mean())
    df['curr_encoded'].fillna(overall_mean_price, inplace=True)
    bins = [0, 50000, 100000, 150000, 200000, float('inf')]
    labels = ['0-50k', '50k-100k', '100k-150k', '150k-200k', '200k+']
    reference_df['km_binned'] = pd.cut(reference_df['Km'], bins=bins, labels=labels)
    print(reference_df['km_binned'])
    print(reference_df.groupby('km_binned')['Price'].mean())
    df['km_binned'] = pd.cut(df['Km'], bins=bins, labels=labels)
    df['km_encoded'] = df['km_binned'].map(reference_df.groupby('km_binned')['Price'].mean())
    print(df['km_binned'])
    # Drop original categorical columns and the Km column
    df = df.drop(['manufactor', 'model', 'Prev_ownership', 'Curr_ownership', 'Gear', 'Engine_type', 'Km', 'km_binned'], axis=1)

    return df
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Print the received form data
        print("Received form data:", request.form)

        manufacture = request.form['manufacture']
        model = request.form['model']
        Year = int(request.form['year'])
        Km = float(request.form['km'])
        Hand = int(request.form['hand'])
        capacity_Engine = float(request.form['capacityEngine'])
        Engine_type = request.form['engineType']
        Gear = request.form['gear']
        Prev_ownership = request.form['previousOwnership']
        Curr_ownership = request.form['currentOwnership']
        Supply_score = request.form.get('supplyScore', -1) #if there is no supply score the defult is -1.
        print(type(Supply_score))
        #create dataframe that contains all the columns like the original dataset.
        input_data = pd.DataFrame({
            'manufactor': [manufacture],
            'model': [model],
            'Year': [Year],
            'Km': [Km],
            'Hand': [Hand],
            'capacity_Engine': [capacity_Engine],
            'Engine_type': [Engine_type],
            'Gear': [Gear],
            'Prev_ownership': [Prev_ownership],
            'Curr_ownership': [Curr_ownership],
            'Supply_score': [Supply_score],
            'Area': [""],
            "City": [""],
            "Pic_num": [""],
            "Cre_date": [""],
            "Repub_date": [""],
            "Color": [""],
            "Description": [""],
            "Test": [""],
            "Price": [0]
        })

        print("Initial Input DataFrame:\n", input_data)
            
        input_data = prepare_data(input_data)
        print("Prepared DataFrame:\n", input_data)

        input_data = encoding(input_data, Train_set)
        print("Encoded DataFrame:\n", input_data)
        input_data = input_data.drop('Price', axis=1)  # Features
        final_features = input_data.values
        print("Final Features:\n", final_features)


        prediction = car_model.predict(final_features)[0] # predict the price with the features
        prediction=round(prediction,0) #The prediction

        if prediction >= 0:
            output_text = f"The car price is: ₪{int(prediction):,}"
        else:
            output_text = "Prediction could not be made."

        debug_info = f"Input Data:\n{input_data}\nFinal Features:\n{final_features}\nPrediction: {prediction}"

    except Exception as e:
        output_text = f"An error occurred: {str(e)}"
        debug_info = str(e)
    # Render the 'index.html' web page and display the prediction result and debug information
    return render_template('index.html', prediction_text=output_text, debug_info=debug_info)

if __name__ == "__main__":
    # Get the port number from the environment variable 'PORT' or use 5000 as the default
    port = int(os.environ.get('PORT', 5000))
    # Start the Flask web server, enabling debug mode and listening on all network interfaces
    app.run(debug=True, host='0.0.0.0', port=port)

# cd flaskCarPrice
# venv\Scripts\activate
# python api.py
#
# http://127.0.0.1:5000/
 
 