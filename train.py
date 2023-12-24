from dotenv import load_dotenv
import os
from fastai.text.all import *

# TODO: Create Dataframe from sql

# Load variables from .env file
load_dotenv()

# # Access the variables
# api_key = os.getenv('API_KEY')
# secret_key = os.getenv('SECRET_KEY')

# query = "Select ee.id,name,expenseTypeId from expense.ExpenseEntries ee,expense.ExpenseEntryToTypeMaps eettm WHERE ee.id=eettm.ExpenseEntryId"


#Train script

df = pd.read_csv("expenses.csv")
df['name'] = df['name'].str.lower()
dls = TextDataLoaders.from_df(df, text_col='name', label_col='expenseTypeId', valid_pct = 1e-10)

learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(4, 5e-2)
learn.fine_tune(4, 5e-2)
learn.fine_tune(4, 1e-3)
learn.fit_one_cycle(10, 1e-3)

learn.export('models/learner.pkl')