import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score

taget_H2 = 'H2_converted_vol%'
taget_CH4 = 'CH4_converted_vol%'
H2_save_path = './H2_regressor_model'
CH4_save_path = './CH4_regressor_model'

data = pd.read_csv("data/H2_basictest_augmented.csv", index_col = 0)
test = pd.read_csv("data/H2_basictest.csv", index_col = 0)

train, val = train_test_split(data, train_size = 0.8, shuffle = True)

train_H2 = train.drop(labels = taget_CH4, axis = 1)
X_train_H2 = train_H2.drop(labels = taget_H2, axis = 1)
Y_train_H2 = train_H2[taget_H2]

train_CH4 = train.drop(labels = taget_H2, axis = 1)
X_train_CH4 = train_CH4.drop(labels = taget_CH4, axis = 1)
Y_train_CH4 = train_CH4[taget_CH4]

true_H2 = val[taget_H2]
true_CH4 = val[taget_CH4]

val = val.drop(taget_H2, axis = 1)  
val = val.drop(taget_CH4, axis = 1)

# param_grid={
#     'learning_rate':[0.01, 0.015], 
#     'max_depth':[0, 10], 
#     'n_estimators':[1000, 1500]
#     }

xgb_params = {
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "max_depth": 10,
    "max_leaves": 100,
    "min_child_weight": 10,
    "subsample": 0.5,
    "colsample_bytree": 0.6,
    "colsample_bylevel": 0.6,
    "gamma": 18,
    "eval_metric": "mape",
    "early_stopping_rounds": 50,
    "random_state": 514
}

H2_Rmodel = XGBRegressor(**xgb_params)

# H2_Rmodel = XGBRegressor()

# H2_grid_search = GridSearchCV(H2_Rmodel, param_grid = param_grid, cv = 5, scoring = 'neg_mean_absolute_percentage_error')
# H2_grid_search.fit(X_train_H2, Y_train_H2)

H2_eval_set = [(val, true_H2)]

CH4_Rmodel = XGBRegressor(**xgb_params)
# CH4_Rmodel = XGBRegressor()

# CH4_grid_search = GridSearchCV(CH4_Rmodel, param_grid = param_grid, cv = 5, scoring = 'neg_mean_absolute_percentage_error')
# CH4_grid_search.fit(X_train_CH4, Y_train_CH4)

CH4_eval_set = [(val, true_CH4)]

# print('best parameters : ', H2_grid_search.best_params_)
# print('best score : ', round(H2_grid_search.best_score_, 4))

# print('best parameters : ', CH4_grid_search.best_params_)
# print('best score : ', round(CH4_grid_search.best_score_, 4))

# best parameters :  {'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 1000}
# best score :  -0.0311
# best parameters :  {'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 1000}
# best score :  -0.0413

H2_Rmodel.fit(X = X_train_H2,
          y = Y_train_H2,
          eval_set=H2_eval_set,
          verbose=True)

CH4_Rmodel.fit(X = X_train_CH4,
          y = Y_train_CH4,
          eval_set=CH4_eval_set,
          verbose=True)

H2_pred = H2_Rmodel.predict(val)
CH4_pred = CH4_Rmodel.predict(val)

H2_mae = mean_absolute_error(true_H2, H2_pred)
H2_mape = mean_absolute_percentage_error(true_H2, H2_pred)

CH4_mae = mean_absolute_error(true_CH4, CH4_pred)
CH4_mape = mean_absolute_percentage_error(true_CH4, CH4_pred)

# 1.4415602259347589 0.031151609973779488
# 1.2765604896885991 0.041467474649272626

print(H2_mae, H2_mape)
print(CH4_mae, CH4_mape)

H2_Rmodel.fit(X = X_train_H2,
          y = Y_train_H2,
          eval_set=H2_eval_set,
          verbose=True)

CH4_Rmodel.fit(X = X_train_CH4,
          y = Y_train_CH4,
          eval_set=CH4_eval_set,
          verbose=True)

X_test = test.drop(labels = taget_H2, axis = 1)
X_test = X_test.drop(labels = taget_CH4, axis = 1)

Y_test_true_H2 = test[taget_H2]
Y_test_true_CH4 = test[taget_CH4]

test_H2_pred = H2_Rmodel.predict(X_test)
test_CH4_pred = CH4_Rmodel.predict(X_test)

test_H2_mae = mean_absolute_error(Y_test_true_H2, test_H2_pred)
test_H2_mape = mean_absolute_percentage_error(Y_test_true_H2, test_H2_pred)

test_CH4_mae = mean_absolute_error(Y_test_true_CH4, test_CH4_pred)
test_CH4_mape = mean_absolute_percentage_error(Y_test_true_CH4, test_CH4_pred)

print(test_H2_mae, test_H2_mape)
print(test_CH4_mae, test_CH4_mape)

H2_Rmodel.save_model('model/H2_Rmodel.dat')
CH4_Rmodel.save_model('model/CH4_Rmodel.dat')

# 0.13709452317312762 0.003067826577590255
# 0.13508969332301315 0.00429995357516302