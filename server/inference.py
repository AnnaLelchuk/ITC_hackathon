import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from yscore import Yscore
from sklearn.ensemble import RandomForestRegressor


with open("../model/xgbr_model.pkl", 'rb') as f:
    model: XGBRegressor = pickle.load(f)

# with open("../model/rfr_model.pkl", 'rb') as f:
#     model: RandomForestRegressor = pickle.load(f)


def format_input(form: pd.DataFrame):
    columns = ['emp_length', 'annual_inc', 'delinq_2yrs', 'mths_since_last_delinq', 'tot_cur_bal',
               'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mort_acc', 'num_actv_bc_tl',
               'home_ownership_MORTGAGE', 'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
               'application_type_Individual', 'application_type_Joint App']
    nominal = ['home_ownership', 'application_type']
    numerical = ["emp_length", "annual_inc", "delinq_2yrs", "mths_since_last_delinq", "tot_cur_bal",
                 "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mort_acc", "num_actv_bc_tl"]
    enum2name = {
        'home_ownership':
            {
                0: "RENT",
                1: "MORTGAGE",
                2: "OWN",
                3: "OTHER"
            },
        'application_type':
            {
                0: 'Individual',
                1: 'Joint App'
            }
    }
    input_dict = {col: 0 for col in columns}

    for feature in numerical:
        input_dict[feature] = int(form[feature].iloc[0])

    for column in nominal:
        feature = column + '_' + enum2name[column][int(form[column])]
        input_dict[feature] = 1

    df = pd.DataFrame(columns=columns)
    df = df.append(input_dict, ignore_index=True)

    return df.to_numpy()


def round_prediction(prediction: float):
    prediction = 10 * np.round(prediction/10)
    return int(prediction)

columns = ['emp_length', 'annual_inc', 'delinq_2yrs', 'mths_since_last_delinq', 'tot_cur_bal',
               'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mort_acc', 'num_actv_bc_tl',
               'home_ownership_MORTGAGE', 'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
               'application_type_Individual', 'application_type_Joint App']
def predict(form: pd.DataFrame):
    form = format_input(form)
    yscore = Yscore("../model/xgbr_model.pkl")
    yscore.feature_weigths(form, columns)
    prediction = model.predict(form)
    prediction = round_prediction(prediction[0])
    return prediction


if __name__ == '__main__':
    print(model)
    test_dict = {
        'emp_length': 0,
        'annual_inc': 49875,
        'delinq_2yrs': 32,
        'mths_since_last_delinq': 9.29,
        'tot_cur_bal': 0,
        'mo_sin_old_rev_tl_op': 37,
        'mo_sin_rcnt_tl': 20,
        'mort_acc': 0,
        'num_actv_bc_tl': 2,
        'home_ownership_MORTGAGE': 0,
        'home_ownership_OTHER': 0,
        'home_ownership_OWN': 0,
        'home_ownership_RENT': 1,
        'application_type_Individual': 0,
        'application_type_Joint App': 1
    }
    test_input = pd.DataFrame(test_dict, index=[0]).to_numpy()
    print(model.predict(test_input)[0])
