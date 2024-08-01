from functions import remove_price_outliers , replace_hebrew_in_model , remove_before_comma , remove_after_word,remove_words , translate_model,normalize_text , fuzzy_match , get_supply_info_dict , fill_missing_supply_scores , replace_values , merge_supply_score_info , fill_from_description , fill_missing_ownership , fill_km , fill_capacity_engine 
import pandas as pd 

#Test_data = pd.read_csv("dataset.csv") ## enter your test data here

def prepare_data(df):
    print("Initial DataFrame in prepare_data:\n", df)
    df.drop_duplicates(inplace=True) # Remove duplicate rows
    print("drop_duplicates in prepare_data:\n", df)
    df = df.drop(['Area','City','Pic_num','Cre_date','Repub_date','Color'],axis = 1) # drop columns that are not required for the prediction
    print("drop(['Area','City','Pic_num','Cre_date','Repub_date','Color'] in prepare_data:\n", df)
    df = replace_values(df) # replace unwanted values
    print(" replace_values in prepare_data:\n", df)
    manufacturer_model_dict = df.groupby('manufactor_GOV')[['model_english', 'Year']].apply(lambda x: list(zip(x['model_english'], x['Year']))).to_dict()
    print(" manufacturer_model_dict in prepare_data:\n", df)
    if(df['Supply_score'] == '' ).any(): # if there is no supply score given, predict the suply score
        print(" Web scraping the supply score, if not found, Predict the supply score... please wait...")
        supply_info_dict, not_found_count = get_supply_info_dict(manufacturer_model_dict) # Returns dictionary of the supply score from the web scraping 
        df = merge_supply_score_info (supply_info_dict,df) # adds to the df the founded supply score
        print(" merge in prepare_data:\n", df)
        df, results, y_test, X_test, best_model_name, best_pipeline = fill_missing_supply_scores(df) # filling the missing values of the supply score by using the best prediction model
        print("fill_missing_supply_scores in prepare_data:\n", df)
        df['Supply_score'] = df['Supply_score_All']
        print("333:\n", df)
        df = df.drop(['Supply_score_All','model_supply'], axis=1)
    print(" 2 in prepare_data:\n", df)
    df.drop('Test', axis=1, inplace=True) # drop 'test' column due to large amount of missing values
    print(" 3 in prepare_data:\n", df)
    df = fill_from_description(df) # fiil missing values if they are in the description
    print(" 4 in prepare_data:\n", df)
    df= fill_missing_ownership(df) # fill missing values in the cuur_ownership and prev_ownership based on the the described logic
    print(" 5 in prepare_data:\n", df)
    df = fill_km (df) # prepare the KM column 
    print(" 6 in prepare_data:\n", df)
    df = df.dropna(axis = 0,subset = ['Engine_type'])
    print(" 7 in prepare_data:\n", df)
    df = fill_capacity_engine(df) # prepare the capacity_Engine column 
    print(" 8 in prepare_data:\n", df)
    df = df.dropna(axis = 0,subset = ['Gear'])
    print(" 9 in prepare_data:\n", df)
    df = df.drop(['Description','manufactor_GOV','model'],axis=1)
    print(" 11 in prepare_data:\n", df)
    df = df.rename(columns={'model_english': 'model'})
    print(" 12 in prepare_data:\n", df)
    df.drop_duplicates(inplace=True)
    print(" 13 in prepare_data:\n", df)
    #df = remove_price_outliers(df) # remove price outliers base on the highest corralted features (uses the heatmap corrolation)
    print("Processed DataFrame in prepare_data:\n", df)
    return df

#df= prepare_data(Test_data)
#df.to_csv('Train_set.csv', index=False, encoding='utf-8-sig')  # 'carTEST.csv' is the name of the file to be created