import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor


with open("model/xgbr_model.pkl", 'rb') as f:
    model: XGBRegressor = pickle.load(f)


def predict(input: np.ndarray):
    prediction = model.predict(input)


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
