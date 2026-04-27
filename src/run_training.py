from data_preporcessing import load_data, select_columns, rename_columns, clean_data
from feature_engineering import prepare_features
from train import split_data, train_model, evaluate_model, save_model,save_predictions


df = load_data("data/raw/world_bank_data_2025.csv")

df = select_columns(df)
rename_columns(df)
df = clean_data(df)

df = prepare_features(df)

train, val, test = split_data(df)

X_train = train.drop(columns=['Inflation_CPI'])
y_train = train['Inflation_CPI']

X_val = val.drop(columns=['Inflation_CPI'])
y_val = val['Inflation_CPI']

X_test = test.drop(columns=['Inflation_CPI'])
y_test = test['Inflation_CPI']

model = train_model(X_train, y_train)

val_mae,_ = evaluate_model(model, X_val, y_val)
test_mae,test_preds = evaluate_model(model,X_test,y_test)

save_model(model, 'models/model_script.joblib')

print("Validation MAE:", val_mae)
print("Test MAE:", test_mae)

save_predictions(test, test_preds)

