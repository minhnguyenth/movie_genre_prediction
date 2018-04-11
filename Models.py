import pandas as pd



data_set =  pd.read_csv("usa_movies_processed.csv")
data_set_json =  pd.read_json("usa_movies_processed.json", lines=True)
temp = 10