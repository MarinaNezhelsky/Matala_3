import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import requests
import pandas as pd
import unicodedata
import difflib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def replace_hebrew_in_model(df):
    for index, row in df.iterrows():
        manufactor = row['manufactor']
        year = str(row['Year'])  # Convert year to string for comparison
        model = row['model']        
        if manufactor in model:
            df.at[index, 'model'] = model.replace(manufactor, '')        
        if year in model:
            df.at[index, 'model'] = model.replace(f'({year})', '') 
    return df


# Transformation functions
def remove_before_comma(s):
    return s.split(',', 1)[-1].lstrip()


def remove_after_word(s):
    if 'CLASS' in s:
        s = s.split('CLASS', 1)[0] + 'CLASS'
    if 'מיטו' in s:
        s = s.split('מיטו', 1)[0] + 'מיטו' 
    index = s.find('דור')
    if index != -1:
        s = s[:index]
    return s


def remove_words(text):
    words_to_remove = [ 'דור 4','חשמלי','החדשה','הדור החדש','חדשה','סקודה']
    for word in words_to_remove:
        text = text.replace(word, '')
    return text.strip()

def translate_model(hebrew_model):
    return car_models_translation.get(hebrew_model, hebrew_model)  # Return English translation or original if not found

# The function clean up and normalize the input text.
def normalize_text(text):
    if text is not None:
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        return text.strip().upper()
    return None

# The function aims to find the closest match for a given model name from a list of model names using a similarity threshold. 
def fuzzy_match(model, models_list, threshold=0.8):
    matches = difflib.get_close_matches(model, models_list, n=1, cutoff=threshold)
    return matches[0] if matches else None

# The scraping function to get supply info
def get_supply_info_dict(model_dict):
    supply_info = {}
    not_found_count = 0
    try:
        url = 'https://data.gov.il/api/3/action/datastore_search?resource_id=5e87a7a1-2f6f-41c1-8aec-7216d52a6cf6'
        records = []
        offset = 0
        limit = 1000  
        iteration_count = 0
        while True:
            iteration_count += 1
            print(f" Page count: {iteration_count}", end='\r')
            response = requests.get(url, params={'offset': offset, 'limit': limit})
            if response.status_code != 200:
                break
            results_page = response.json()
            new_records = results_page['result']['records']
            if not new_records:
                break  # Exit the loop if there are no more records
            records.extend(new_records)
            offset += limit  # Fetch the next page of results

        # Convert to DataFrame
        df_records = pd.DataFrame(records)
        df_records['kinuy_mishari'] = df_records['kinuy_mishari'].apply(normalize_text)
        df_records['mispar_rechavim_pailim'] = pd.to_numeric(df_records['mispar_rechavim_pailim'], errors='coerce').fillna(0).astype(int)

        for manufacturer, models_years in model_dict.items():
            for model, year in models_years:
                normalized_model = normalize_text(model)
                filtered_records = df_records[
                    (df_records['tozar'] == manufacturer) &
                    (df_records['shnat_yitzur'] == year) &
                    (df_records['kinuy_mishari'] == normalized_model)
                ]
                if filtered_records.empty:
                    # If no records found, set supply score to 0
                    supply_info[(manufacturer, model, year)] = 0
                else:
                    closest_match = fuzzy_match(normalized_model, filtered_records['kinuy_mishari'].tolist())
                    if closest_match:
                        total_supply_score = filtered_records[
                            filtered_records['kinuy_mishari'] == closest_match
                        ]['mispar_rechavim_pailim'].sum()
                        supply_info[(manufacturer, model, year)] = total_supply_score
                    else:
                        # If no close match found, set supply score to 0
                        supply_info[(manufacturer, model, year)] = 0

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return supply_info, not_found_count

def fill_missing_supply_scores(df):
    df_train = df.dropna(subset=['Supply_score_All'])
    X = df_train[['manufactor_GOV', 'model', 'Year']]  # Features
    y = df_train['Supply_score_All']  # Target variable
    # Split the data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # One-hot encode categorical variables
    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'),['manufactor_GOV','model']),],remainder='passthrough')

    # Models to evaluate
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'RandomForestRegressor': RandomForestRegressor()}
    results = {}
    best_pipeline = None
    
    # Create a pipeline for each model
    for model_name, model in models.items():
        pipeline = Pipeline([('preprocessor', preprocessor),('regressor', model)])
        pipeline.fit(X, y)
        # Predict on the test set
        y_pred = pipeline.predict(X)
        y_pred = np.maximum(y_pred, 0)
        # Calculate Mean Squared Error
        mse = mean_squared_error(y, y_pred)
        results[model_name] = mse
        if best_pipeline is None or mse < results[min(results, key=results.get)]:
            best_pipeline = pipeline

    # Find model with lowest MSE on training data
    best_model_name = min(results, key=results.get)
    # Predict missing values in df using the best model
    X_missing = df.loc[df['Supply_score_All'].isnull(), ['manufactor_GOV', 'model', 'Year']]

    if not X_missing.empty:
        # Predict missing values
        predicted_values = best_pipeline.predict(X_missing)
        # Fill predicted values back into df
        rounded_values = np.round(predicted_values)
        df.loc[df['Supply_score_All'].isnull(), 'Supply_score_All'] = rounded_values
    
    return df, results, y, X, best_model_name, best_pipeline



def replace_values(df):
    # manufactor column
    df['manufactor'] = df['manufactor'].replace('Lexsus','לקסוס')
    
    # model column
    df['model'] = df['model'].str.upper()
    df = replace_hebrew_in_model(df)
    df['model'] = df['model'].str.replace('`', "'")
    df['model'] = df['model'].str.replace('קאונטרימן', "קאנטרימן")
    df['model'] = df['model'].apply(remove_before_comma)
    df['model'] = df['model'].apply(remove_after_word)
    df['model'] = df['model'].apply(remove_words)
    df['model'] = df['model'].str.strip()
    
    # Gear column
    df['Gear'] = df['Gear'].replace('אוטומט','אוטומטית')
    df['Gear'] = df['Gear'].replace('לא מוגדר',np.nan)
    df['Gear'] = df['Gear'].replace('None',np.nan)
    
    # capacity_Engine column
    df['capacity_Engine'] = df['capacity_Engine'].replace('nan', np.nan)
    df['capacity_Engine'] = df['capacity_Engine'].replace('לא מוגדר', np.nan)
    df['capacity_Engine'] = df['capacity_Engine'].replace('None', np.nan)
    df['capacity_Engine'] = df['capacity_Engine'].replace(',', '', regex=True)
    df['capacity_Engine'] = df['capacity_Engine'].astype(float).astype('Int64')
    
    # Missing values in supply_score 
    df['model_english'] = df['model'].apply(translate_model)
    df['manufactor_GOV'] = df['manufactor']
    df['manufactor_GOV'] = df['manufactor_GOV'].replace('וולוו','וולבו')
    df['manufactor_GOV'] = df['manufactor_GOV'].replace('מאזדה','מזדה')
    df['manufactor_GOV'] = df['manufactor_GOV'].replace('ב.מ.וו','ב מ וו')
    df['manufactor_GOV'] = df['manufactor_GOV'].replace('מיני','ב מ וו')
    df.loc[(df['manufactor'] == 'מרצדס') & df['Description'].str.contains('S 600', na=False), 'model_english'] = 'S600'
    df.loc[(df['manufactor'] == 'מרצדס') & df['Description'].str.contains('c200', na=False), 'model_english'] = 'C200'
    df.loc[(df['manufactor'] == 'מרצדס') & df['Description'].str.contains('e350', na=False), 'model_english'] = 'E350'
    df.loc[(df['manufactor'] == 'מרצדס') & df['Description'].str.contains('s 550', na=False), 'model_english'] = 'S550'
    df.loc[(df['manufactor'] == 'מרצדס') & df['Description'].str.contains('ml320', na=False), 'model_english'] = 'ML320'
    df.loc[(df['model_english'] == 'SUPERB') & (df['Year'] > 2015), 'model_english'] = 'NEW SUPERB'
    
    # Curr_ownership
    df['Curr_ownership'] = df['Curr_ownership'].replace('לא מוגדר',np.nan)
    df['Curr_ownership'] = df['Curr_ownership'].replace('None',np.nan)
    
    # Prev_ownership 
    df['Prev_ownership'] = df['Prev_ownership'].replace('לא מוגדר',np.nan)
    df['Prev_ownership'] = df['Prev_ownership'].replace('None',np.nan)
    
    # Km column 
    df['Km'] = df['Km'].replace(['לא מוגדר', 'None'], np.nan)
    df['Km'] = df['Km'].astype(str).str.replace(',', '').astype(float)
    
    # Engine_type
    df['Engine_type'] = df['Engine_type'].replace('None',np.nan)
    df['Engine_type'] = df['Engine_type'].replace('לא מוגדר',np.nan)
    df['Engine_type'] = df['Engine_type'].replace('היבריד','היברידי')
    
    

    return df

def merge_supply_score_info(supply_info_dict, df):
    # Initialize lists
    manufacturers = []
    models = []
    years = []
    supply_scores = []

    # Iterate over the dictionary
    for (manufacturer, model, year), supply_score in supply_info_dict.items():
        manufacturers.append(manufacturer)
        models.append(model)
        years.append(year)
        supply_scores.append(supply_score)

    # Create DataFrame from lists
    supply_info_df = pd.DataFrame({
        'manufactor_GOV': manufacturers,
        'model': models,
        'Year': years,
        'Supply_score_All': supply_scores
    })

    # Perform the merge with suffixes
    merged_df = df.merge(
        supply_info_df[['manufactor_GOV', 'model', 'Year', 'Supply_score_All']],
        left_on=['manufactor_GOV', 'model_english', 'Year'],
        right_on=['manufactor_GOV', 'model', 'Year'],
        how='left',
        suffixes=('', '_supply')
    )

    return merged_df

def fill_from_description (df):
    df.loc[pd.isna(df['Curr_ownership']) & df['Description'].str.contains('פרטית', na=False), 'Curr_ownership'] = 'פרטית'
    df.loc[pd.isna(df['Curr_ownership']) & df['Description'].str.contains('ליסינג', na=False), 'Curr_ownership'] = 'ליסינג'
    df.loc[pd.isna(df['Curr_ownership']) & df['Description'].str.contains('השכרה', na=False), 'Curr_ownership'] = 'השכרה'
    df.loc[pd.isna(df['Curr_ownership']) & df['Description'].str.contains('חברה', na=False), 'Curr_ownership'] = 'חברה'
    df.loc[pd.isna(df['Curr_ownership']) & df['Description'].str.contains('מונית', na=False), 'Curr_ownership'] = 'מונית'
    df.loc[pd.isna(df['Curr_ownership']) & df['Description'].str.contains('ממשלתי', na=False), 'Curr_ownership'] = 'ממשלתי'
    df.loc[pd.isna(df['Gear']) & df['Description'].str.contains('רובוטית', na=False), 'Gear'] = 'רובוטית'
    df.loc[pd.isna(df['Gear']) & df['Description'].str.contains('טיפטרוניק', na=False), 'Gear'] = 'טיפטרוניק'
    df.loc[pd.isna(df['Gear']) & df['Description'].str.contains('ידנית', na=False), 'Gear'] = 'ידנית'
    df.loc[pd.isna(df['Gear']) & df['Description'].str.contains('אוטומטית', na=False), 'Gear'] = 'אוטומטית'
    df.loc[pd.isna(df['Gear']) & df['Description'].str.contains('אוטומטית', na=False), 'Gear'] = 'אוטומט'    
    return df
    
def fill_missing_ownership (df):
    df['Curr_ownership'].fillna(value='פרטית', inplace=True)
    df.loc[df['Hand'] == 1, 'Prev_ownership'] = 'חדש' 
    df.loc[pd.isna(df['Prev_ownership']) , 'Prev_ownership'] =df['Curr_ownership']    
    return df
    
def fill_km (df):
    # for values under 1000, multiply by 1000 (195K=195,000)
    df['Km'] = df['Km'].apply(lambda x: x * 1000 if pd.notna(x) and x < 1000 else x)
    # Replacing values under 10K that can be mistake to nan
    df['Km'] = df['Km'].apply(lambda x: np.nan if pd.notna(x) and x < 10000 else x)
    # Replacing outliers with NaN based on year (more or less than 3*STD)
    mean_km = df['Km'].mean()
    std_km = df['Km'].std()
    UCL = mean_km + 3 * std_km
    LCL = max(0,mean_km - 3 * std_km) 
    df['Km'] = df['Km'].apply(lambda x: np.nan if pd.notna(x) and (x > UCL or x < LCL) else x)
    grouped_data = df.groupby(['Year', 'Hand'])['Km'].median().reset_index()
    df = pd.merge(df, grouped_data, on=['Year', 'Hand'], how='left', suffixes=('', '_median'))
    df['Km'].fillna(df['Km_median'], inplace=True)
    grouped_data2 = df.groupby('Year')['Km'].median().reset_index()
    df = pd.merge(df, grouped_data2, on=['Year'], how='left', suffixes=('', '_median2'))
    df['Km'].fillna(df['Km_median2'], inplace=True)
    df.drop(columns=['Km_median','Km_median2'], inplace=True)
    df['Km'] = df['Km'].astype(str).str.replace(',', '').astype(float)
    df = df.dropna(axis = 0,subset = ['Km'])
    return df

def fill_capacity_engine (df):
    df['capacity_Engine'] = df['capacity_Engine'].apply(lambda x: x / 10 if pd.notna(x) and x > 10000 else x)
    # Step 1: Calculate the median capacity_Engine for each model where possible
    df['mean_capacity'] = df.groupby('model_english')['capacity_Engine'].transform(lambda x: x.mean() if x.notnull().any() else np.nan)
    # Step 2: Calculate the standard deviation for each model where possible
    df['std_capacity'] = df.groupby('model_english')['capacity_Engine'].transform(lambda x: x.std() if x.notnull().any() else np.nan)
    df['std_capacity']=df['std_capacity'].replace(np.nan, 0)
    # Step 3: Adjust capacity_Engine values where possible
    condition = (df['capacity_Engine'] < df['mean_capacity'] - 3 * df['std_capacity']) | \
                (df['capacity_Engine'] > df['mean_capacity'] + 3 * df['std_capacity']) | \
                (df['capacity_Engine'] < 800)
    filtered_rows = df[condition]
    filtered_indices = filtered_rows.index
    df.loc[filtered_indices, 'capacity_Engine'] = np.nan
    # Group by 'model' and calculate the median
    df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'], errors='coerce')
    grouped_data = df.groupby(['model'])['capacity_Engine'].median().reset_index()
    grouped_data.rename(columns={'capacity_Engine': 'capacity_Engine_median'}, inplace=True)
    # Merge to get the median value back into the original DataFrame
    df = pd.merge(df, grouped_data, on='model', how='left')
    # Fill NaN values in 'capacity_Engine' with the median values
    df['capacity_Engine'].fillna(df['capacity_Engine_median'], inplace=True)
    # Drop the temporary median column
    df.drop(columns=['capacity_Engine_median','std_capacity','mean_capacity'], inplace=True)
    # Drop rows where 'capacity_Engine' is still NaN
    df.dropna(axis=0, subset=['capacity_Engine'], inplace=True)
    return df

def remove_price_outliers(df):
    # Calculate the mean and standard deviation price for each model
    df['model_mean_price'] = df.groupby('model')['Price'].transform(lambda x: x.mean() if x.notnull().any() else np.nan)
    df['model_std_Price'] = df.groupby('model')['Price'].transform(lambda x: x.std() if x.notnull().any() else np.nan)
    df['model_std_Price'] = df['model_std_Price'].replace(np.nan, 0)
    
    # Identify outliers based on model price
    condition_model = (df['Price'] < df['model_mean_price'] - 2 * df['model_std_Price']) | \
                      (df['Price'] > df['model_mean_price'] + 2 * df['model_std_Price'])
    outliers_rows_by_model = df[condition_model]
    
    # Calculate the mean and standard deviation price for each year
    df['year_mean_price'] = df.groupby('Year')['Price'].transform(lambda x: x.mean() if x.notnull().any() else np.nan)
    df['year_std_Price'] = df.groupby('Year')['Price'].transform(lambda x: x.std() if x.notnull().any() else np.nan)
    df['year_std_Price'] = df['year_std_Price'].replace(np.nan, 0)
    
    # Identify outliers based on year price
    condition_year = (df['Price'] < df['year_mean_price'] - 2 * df['year_std_Price']) | \
                     (df['Price'] > df['year_mean_price'] + 2 * df['year_std_Price'])
    outliers_rows_by_year = df[condition_year]
    
    # Find common outliers in both model and year conditions
    common_outliers = outliers_rows_by_model.index.intersection(outliers_rows_by_year.index)
    common_outliers_df = df.loc[common_outliers]
    
    # Drop common outliers from the original dataframe
    df = df.drop(common_outliers_df.index)
    
    # Drop temporary columns and rows with missing 'Price'
    df.drop(columns=['year_mean_price', 'year_std_Price', 'model_std_Price', 'model_mean_price'], inplace=True)
    df.dropna(subset=['Price'], inplace=True)
    
    return df

car_models_translation = {
    # Audi
    "A1": "A1","A2": "A2","A3": "A3","A4": "A4","A5": "A5","A6": "A6","A7": "A7","A8": "A8","Q2": "Q2","Q3": "Q3","Q5": "Q5","Q7": "Q7","Q8": "Q8","TT": "TT","R8": "R8","RS3": "RS3","RS4": "RS4","RS5": "RS5 COUPE","RS6": "RS6","RS7": "RS7","S3": "S3","S4": "S4","S5": "S5","S6": "S6","S7": "S7","S8": "S8","ALL ROAD":"ALLROAD",

    # Opel
    "אסטרה": "ASTRA","קורסה": "CORSA","אינסיגניה": "INSIGNIA","מריבה": "MERIVA","מוקה": "MOKKA","מוקה X": "MOKKA - X","אדם": "ADAM","זאפירה": "ZAFIRA","קארל": "KARL","אמפרה": "AMPERA","מונזה": "MONZA","קפיטן": "KAPITAN","קומודור": "COMMODORE","קלברא": "CALIBRA",

    # Alfa Romeo
    "ג'וליה": "GIULIA","סטלביו": "STELVIO","מיטו": "ALFA MITO","ג'ולייטה": "ALFA GIULIETTA","4C": "4C","8C": "8C","אלפאסוד": "ALFASUD","אלפאספרינט": "ALFASPRINT","בררה": "BRERA","ספיידר": "SPIDER","145": "145","146": "146","147": "147","155": "155","156": "156","159": "159","164": "164","166": "166",

    # BMW
    "X1": "X1", "X2": "X2", "X3": "X3", "X4": "X4", "X5": "X5", "X6": "X6", "X7": "X7", "Z1": "Z1", "Z3": "Z3", "Z4": "Z4", "i3": "I3", "i8": "I8", "325": "325I", "525": "525I", "523": "523I", "120": "120I", "530": "530I", "318": "318I", "316": "316I",

    #Chevrolet
    "אוואו": "AVEO", "קרוז": "CRUZE", "מאליבו": "MALIBU", "אימפלה": "IMPALA", "ספארק": "SPARK", "סוניק": "SONIC", "קפטיבה": "CAPTIVA", "אורלנדו": "ORLANDO", "טראוורס": "TRAVERSE", "טאהו": "TAHOE", "סילברדו": "SILVERADO", "אפלנדר": "UPLANDER", "קמרו": "CAMARO", "קורסיקה": "CORSICA", "קוואלט": "CAVALIER","קורבט": "CORVETTE", "נובה": "NOVA", "בלייזר": "BLAZER", "אקווינוקס": "EQUINOX", "סוברבן": "SUBURBAN", "וולט": "VOLT", "בולט": "BOLT", "ברלי": "BERETTA", "לומינה": "LUMINA", "מונטה קרלו": "MONTE CARLO", "טראקס":"TRAX",
   

    # Daihatsu
    "סיריון": "SIRION", "טריוס": "TERIOS", "מאטריה": "MATERIA", "גרנד מובר": "GRAND MOVE", "קורה": "COPEN", "מירא": "MIRA", "יונוס": "YRV", "אפלאוס": "APPLAUSE", "פיירי": "CHARADE", "אסקורט": "ESCOORT", "פלטן": "FELLOW", "רוקי": "ROCKY", "פיירי": "CHARADE", "שרמנט":"CHARMANT",

    # Honda
    "סיוויק": "CIVIC", "סיוויק הייבריד": "CIVIC", "סיוויק האצ'בק": "CIVIC", "סיוויק סדאן": "CIVIC", "האצ'בק": "CIVIC", "אקורד": "ACCORD", "CR-V": "CR-V", "ג'אז": "JAZZ", "ג'אז הייבריד": "JAZZ", "HR-V": "HR-V", "פרילוד": "PRELUDE", "אינסייט": "HONDA INSIGHT", "פיילוט": "PILOT", "אג'": "EDGE", "אלמנט": "ELEMENT", "קרוסטור": "CROSSTOUR", "פיט": "FIT", "הונדה E": "HONDA E", "אודסיי": "ODYSSEY", "לג'נד": "LEGEND",

    # Volvo
    "S60": "S60", "S90": "S90", "V40": "V40", "V60": "V60", "V90": "V90", "XC40": "XC40", "XC60": "XC60", "XC90": "XC90", "850": "850", "940": "940", "960": "960", "C30": "C30", "C70": "C70", "S40": "S40", "S70": "S70", "S80": "S80", "V50": "V50", "V70": "V70",
    
    # Toyota
    "קורולה": "COROLLA", "קאמרי": "CAMRY", "רב 4": "RAV4", "יאריס": "YARIS", "פריוס": "PRIUS", "לנד קרוזר": "LAND CRUISER", "היילקס": "HILUX", "סופרה": "SUPRA", "אוונסיס": "AVENSIS", "אונסיס": "AVENSIS", "סיינה": "SIENNA", "סליקה": "CELICA", "ורסו": "VERSO", "טונדרה": "TUNDRA", "טקומה": "TACOMA", "ונזה": "VENZA", "CH-R": "C-HR", "אוריס": "AURIS", "קלוגה": "KLUGER", "ונצה": "VENZA", "מיראי": "MIRAI", "ספייס": "SPACE VERSO", "אייגו":"AYGO",

    # Tesla
    "מודל 3": "MODEL 3", "מודל S": "MODEL S", "מודל X": "MODEL X", "מודל Y": "MODEL Y", "רודסטר": "ROADSTER", "סייברטראק": "CYBERTRUCK", "סמי": "SEMI",

    # Jaguar
    "XF": "XF", "F-TYPE": "F-TYPE", "XE": "XE", "F-PACE": "F-PACE", "E-PACE": "E-PACE", "XJ": "XJ", "I-PACE": "I-PACE", "XK": "XK", "S-TYPE": "S-TYPE", "X-TYPE": "X-TYPE",

    # Hyundai
    "i10": "I10", "i20": "I20", "i30": "I30", "I30CW": "I30", "i40": "I40", "טוסון": "TUCSON", "סנטה פה": "SANTA FE", "קונה": "KONA", "אלנטרה": "ELANTRA", "אקסנט": "ACCENT", "סונטה": "SONATA", "פאליסייד": "PALISADE", "בלנו": "BALENO", "וולוסטר": "VELOSTER", "ולוסטר": "VELOSTER", "אסנט": "ASCENT", "איוניק": "IONIQ HYBRID", "טרג'ט":"TRAJET",
    
    # Kia
    "ריו": "RIO", "פיקנטו": "PICANTO", "קרניבל": "CARNIVAL", "סטוניק": "STONIC", "ספיה": "SEPHIA", "פורטה": "FORTE", "סראטו": "CERATO", "נירו": "NIRO", "נירו EV": "NIRO EV", "נירו PHEV": "NIRO PHEV", "סול": "SOUL", "סיד": "CEED", "פרייד":"PRIDE",
    
    # Lexus
    "NX": "NX", "RX": "RX", "ES": "ES", "GS": "GS", "IS": "IS", "LS": "LS", "LC": "LC", "LX": "LX", "UX": "UX", "SC": "SC", "IS300H": "LEXUS IS300H", "IS250":  "LEXUS IS250", "GS300": "LEXUS GS300", "CT200H": "LEXUS CT200H",

    # Mazda
    "3": "MAZDA 3", "6": "MAZDA 6", "CX-3": "CX-3", "CX-5": "CX-5", "CX-7": "CX-7", "CX-9": "CX-9", "MX-5": "MX-5", "2": "MAZDA 2", "5": "MAZDA 5", "קפלה": "CAPELLA", "רוקי": "ROCKY", "אטנזה": "ATENZA", "סבנה": "SAVANNA", "לנטיס": "LANTIS",

    # Mini
    "קופר": "COOPER", "קאנטרימן": "COUNTRYMAN", "קלאבמן": "CLUBMAN", "פייסמן": "PACEMAN", "רודסטר": "ROADSTER", "קופה": "COUPE", "ג'ון קופר וורקס": "JOHN COOPER WORKS",

    # Mitsubishi
    "אאוטלנדר": "OUTLANDER", "ASX": "ASX", "לנסר": "LANCER", "לנסר ה": "LANCER", "פאג'רו": "PAJERO", "ספייס סטאר": "SPACE STAR", "אקאליפס קרוס": "ECLIPSE CROSS", "מיראז'": "MIRAGE", "גרנדיס": "GRANDIS", "קולט": "COLT", "דיאמנטה": "DIAMANTE", "לנסר אבולושן": "LANCER EVOLUTION", "לנסר ספורטבק":  "LANCER SPORTBAC", "מונטרו": "MONTERO", "פג'רו": "PAJERO", "סיגמה": "SIGMA", "גאלנט": "GALANT", "אטראז'": "ATTRAGE", "אקליפס": "ECLIPSE CROSS",

    # Mercedes
    "A-CLASS": "A-CLASS", "B-CLASS": "B-CLASS", "C-CLASS": "C-CLASS", "E-CLASS": "E CLASS", "S-CLASS": "S-CLASS", "CLA": "CLA", "GLA": "GLA", "GLC": "GLC", "GLE": "GLE", "GLS": "GLS", "G-CLASS": "G CLASS", "SL": "SL", "SLK": "SLK", "CLK": "CLK", "ML": "ML", "AMG GT": "AMG GT", "EQC": "EQC", "SLC": "SLC", "V- CLASS":"V CLASS",

    # Nissan
    "קשקאי": "QASHQAI", "מיקרה": "MICRA", "ג'וק JUKE": "JUKE", "נבארה": "NAVARA", "אקסטרייל": "X-TRAIL", "ליפ": "LEAF", "פטרול": "PATROL", "סאני": "SUNNY", "אלמרה": "ALMERA", "מקסימה": "MAXIMA", "GT-R": "GT-R", "350Z": "350Z", "370Z": "370Z", "קובוסטר": "CUBE", "פרונטר": "FRONTIER", "מורנו": "MURANO", "ארמדה": "ARMADA", "תאנה": "TITAN", "סנטרא": "SENTRA", "רוג'": "ROGUE", "טרנו": "TERRANO", "קלס": "KICKS", "אלתימה":"ALTIMA", "סנטרה":"SENTRA", "נוט":"NOTE", "פרימרה":"PRIMERA",

    # Subaru
    "פורסטר": "FORESTER", "אימפרזה": "IMPREZA", "אאוטבק": "OUTBACK", "XV": "XV", "BRZ": "BRZ", "לגאסי": "LEGACY", "טרייבקה": "TRIBECA", "לברג": "LEVORG", "באחה": "BAJA", "ג'סטי": "JUSTY", "באז": "B9 TRIBECA", "אלקס": "ALCYONE", "אנקס": "ANNEX", "לאונה":"LEON",

    # Suzuki
    "סוויפט": "SWIFT", "ויטארה": "VITARA", "SX4": "SX4", "סלריו": "CELERIO", "איגניס": "IGNIS", "בלנו": "BALENO", "ג'ימני": "JIMNY", "ספלאש": "SPLASH", "אלטו": "ALTO", "קיזאשי": "KIZASHI", "אסקודו": "ESCUDO", "קרימן": "CARRY", "פאן קארי": "FUN CARRY", "ספאזיו": "SPACIO", "וואגון אר": "WAGON R", "מייטי בוי": "MIGHTY BOY", "קרוסאובר": "CROSSOVER", "SX4 קרוסאובר": "SX4 CROSSOVER", "סדן" : "SX4",

    # Seat
    "איביזה": "IBIZA", "לאון": "LEON", "ארונה": "ARONA", "אטקה": "ATECA", "טאראקו": "TARRACO", "אלתאה": "ALTEA", "אלמברה": "ALHAMBRA", "מיי": "MII","קורדובה": "CORDOBA",  "טולדו": "TOLEDO",  "מרבלה": "MARBELLA",  "מלגה": "MALAGA",  "רטמו": "RITMO",  "פנדה": "PANDA",  "אינקה": "INCA",  "פורמנטור": "FORMENTOR",

    # Citroën
    "C3": "C3", "C5": "C5", "C4 קקטוס": "C4 CACTUS", "C4 פיקאסו": "C4 PICASSO", "C1": "C1", "ברלינגו": "BERLINGO", "DS3": "DS3", "DS4": "DS4", "DS5": "DS5", "C2": "C2", "C6": "C6", "ק.אס.אקס": "C-CROSSER", "ZX": "ZX", "קסנטיה": "XANTIA", "סאקסו": "SAXO", "ק.אס.קייפ": "C-CAMPER",

    # Skoda
    "פאביה": "FABIA", "פאביה ספייס":"FABIA SPACE", "אוקטביה": "OCTAVIA", "אוקטביה RS":"OCTAVIA RS", "אוקטביה ספייס": "OCTAVIA SPACE", "סופרב": "SUPERB", "קאדיאק": "KODIAQ", "קאמיק": "KAMIQ", "קארוק": "KAROQ", "ראפיד": "RAPID", "סיטיגו / CITYGO": "CITIGO", "ייטי": "YETI", "פלישיה": "FELICIA", "פאבוריט": "FAVORIT", "אוסטין": "OCTAVIA TOUR", "פורמן": "FORMAN", "סקאוט": "SCOUT", "רומסטר": "ROOMSTER",
    
    # Volkswagen
    "פולו": "POLO", "גולף": "GOLF", "פאסאט": "PASSAT", "טיגואן": "TIGUAN", "טוראן": "TOURAN", "שירוקו": "SCIROCCO", "ארטיאון": "ARTEON", "פאיטון": "PHAETON", "באבילי": "BEETLE", "איד.4": "ID.4", "טוראן": "TOUAREG", "ג'טה": "JETTA", "פולקאן": "VOLCANO", "שירוקו": "SCI",

    # Ford
    "פוקוס": "FOCUS", "פיאסטה": "FIESTA", "מונדיאו": "MONDEO", "קוגה": "KUGA", "אקספלורר": "EXPLORER", "אסקייפ": "ESCAPE", "גלאקסי": "GALAXY", "טורנאו קונקט": "TOURNEO CONNECT", "פומה": "PUMA", "אדג'": "EDGE", "מוסטנג": "MUSTANG", "אקוספורט": "ECOSPORT", "ב-מקס": "B-MAX", "סי-מקס": "C-MAX", "רנג'ר": "RANGER", "פלקס": "FLEX", "אס-מקס": "S MAX",

    # Fiat
    "פנדה": "PANDA", "פונטו": "PUNTO", "500": "500", "500L": "500L", "500X": "500X", "טיפו": "TIPO", "בראבו": "BRAVO", "בראבה": "BRAVA", "סיינה": "SIENA", "אונו": "UNO", "קרומא": "CROMA", "מולטיפלה": "MULTIPLA", "דוראלדו": "DUCATO", "קובו": "QUBO", "פאליו": "PALIO", "ריטמו": "RITMO", "דובלו": "DOBLO",

    # Peugeot
    "208": "208", "2008": "2008", "3008": "3008", "308": "308", "508": "508", "5008": "5008", "4008": "4008", "206": "206", "306": "306", "406": "406", "607": "607", "807": "807", "PARTNER": "PARTNER", "RCZ": "RCZ", "BX": "BX", "305": "305", "405": "405", "505": "505", "605": "605",
    
    #Renualet
    "קליאו": "CLIO", "מגאן": "MEGANE", "קפצ'ור": "CAPTUR", "קולאוס": "KOLEOS", "טווינגו": "TWINGO", "פלואנס": "FLUENCE", "לאטיטוד": "LATITUDE", "סקאלה": "SCALA", "טרפיק": "TRAFIC", "קנגו": "KANGOO", "מאסטר": "MASTER", "לוגאן": "LOGAN", "דאסטר": "DUSTER", "זואי": "ZOE", "סניק": "SCENIC", "גראנד סניק": "GRAND SCENIC", "גרנד סניק":"GRAND SCENIC", "אילקטריק": "ELECTRIC", "טאליסמן": "TALISMAN", "אלפין": "ALPINE", "אוויאון": "AVANTIME", "אוויאטור": "AVANTIME", "אגו": "KADJAR", "אספאס": "ESPACE", "לאגונה": "LAGUNA", "ספייסר": "SPIDER", "פלואנס Z.E.": "FLUENCE Z.E.",
    
    #Volchvagen
    "גולף": "GOLF", "גולף פלוס":"GOLF PLUS", "פולו": "POLO", "פאסאט": "PASSAT", "טיגואן": "TIGUAN", "טוארג": "TOUAREG", "ג'טה": "JETTA", "ביטל": "BEETLE", "שירוקו": "SCIROCCO", "שיראן": "SHARAN", "אמארוק": "AMAROK", "אפליקטה": "ARTEON", "אידי 3": "ID.3", "אידי 4": "ID.4", "אידי 6": "ID.6", "אפ!": "UP!", "קרבל": "CARAVELLE", "טרנספורטר": "TRANSPORTER", "מולטיוואן": "MULTIVAN", "קראוון": "CADDY", "פולו GTI": "POLO GTI", "גולף GTI": "GOLF GTI", "גולף R": "GOLF R", "פאסאט VR6": "PASSAT VR6", "פאסאט CC": "PASSAT CC", "חיפושית": "BEETLE", 

    "וויאג'ר":"VOYAGER",
    # Jaguar
    "XF": "XF","XJ": "XJ","F-PACE": "F-PACE","E-PACE": "E-PACE","I-PACE": "I-PACE","F-TYPE": "F-TYPE","XE": "XE","XK": "XK",

    # Tesla
    "Model S": "MODEL S","Model 3": "MODEL 3","Model X": "MODEL X","Model Y": "MODEL Y","Roadster": "ROADSTER","Cybertruck": "CYBERTRUCK"
}

