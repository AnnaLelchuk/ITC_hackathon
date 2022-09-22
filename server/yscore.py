import numpy as np
import pandas as pd
from shap import TreeExplainer
import pickle
from xgboost import XGBRegressor
import math
import testing as tst


class Yscore:
    columns = ['emp_length', 'annual_inc', 'delinq_2yrs', 'mths_since_last_delinq', 'tot_cur_bal',
               'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mort_acc', 'num_actv_bc_tl',
               'home_ownership_MORTGAGE', 'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
               'application_type_Individual', 'application_type_Joint App']
    nominal = ['home_ownership', 'application_type']
    numerical = ["emp_length", "annual_inc", "delinq_2yrs", "mths_since_last_delinq", "tot_cur_bal",
                 "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mort_acc", "num_actv_bc_tl"]

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        """
        Loads the model from a file
        :return: xgboost regression model
        """
        with open(self.model_path, 'rb') as f:
            xgb_model: XGBRegressor = pickle.load(f)
        return xgb_model

    def format_input(self, form: pd.DataFrame):
        """
        Formats the received form to an input expected by the model.
        :param form: user input form
        :return: numpy array to put in the model
        """
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
        input_dict = {col: 0 for col in self.columns}

        for feature in self.numerical:
            input_dict[feature] = int(form[feature].iloc[0])

        for column in self.nominal:
            feature = column + '_' + enum2name[column][int(form[column])]
            input_dict[feature] = 1

        df = pd.DataFrame(columns=self.columns)
        df = df.append(input_dict, ignore_index=True)

        return df.to_numpy()

    @staticmethod
    def round_dozen(number: float):
        "Rounds the given number to the dozens"
        number = 10 * np.round(number / 10)
        return int(number)

    def feature_weigths(self, sample: pd.DataFrame, top_feat_num=2):
        """
        :param sample: numpy array with user's data
        :param top_feat_num: number of top features you want to see impacting neg and pos the score
        :return: dictionary with top impacting features and their weight
        """
        model_input = self.format_input(sample)

        score = self.model.predict(model_input)[0]

        explainer = TreeExplainer(self.model)
        shap_values = explainer.shap_values(model_input)

        feat_imp = pd.DataFrame(np.around(shap_values, 5))
        df_imp = pd.concat((pd.DataFrame(self.columns), feat_imp.T), axis=1)
        df_imp.columns = ['feature', 'importance']
        df_imp.sort_values(by='importance', inplace=True, ascending=False)
        df_imp = pd.concat((df_imp.head(top_feat_num), df_imp.tail(top_feat_num)), axis=0)
        result = dict(zip(df_imp.feature, df_imp.importance))
        return self.round_dozen(score), result

    def improve_score(self, sample: pd.DataFrame):
        new_params_dict = {}  # dictionary of updated personal parameters
        current_params_dict = {}

        data = sample.astype(int)
        data_new = data.copy()

        # changing all features that make sense to be changed and writing to dictionary:
        if data['emp_length'][0] < 10:
            data_new['emp_length'] += 1
            current_params_dict['emp_length'] = data['emp_length'][0]
            new_params_dict['emp_length'] = data_new['emp_length'][0]

        if data['annual_inc'][0] < 6:
            data_new['annual_inc'] += 1
            current_params_dict['annual_inc'] = data['annual_inc'][0]
            new_params_dict['annual_inc'] = data_new['annual_inc'][0]


        if (data['mths_since_last_delinq'][0] != -1) & (data['mths_since_last_delinq'][0] < 4):
            data_new['mths_since_last_delinq'] += 1
            current_params_dict['mths_since_last_delinq'] = data['mths_since_last_delinq'][0]
            new_params_dict['mths_since_last_delinq'] = data_new['mths_since_last_delinq'][0]

        if data['tot_cur_bal'][0] < 9:
            data_new['tot_cur_bal'] += 1
            current_params_dict['tot_cur_bal'] = data['tot_cur_bal'][0]
            new_params_dict['tot_cur_bal'] = data_new['tot_cur_bal'][0]

        if data['mort_acc'][0] == 0:
            data_new['mort_acc'] = 1
            current_params_dict['mort_acc'] = data['mort_acc'][0]
            new_params_dict['mort_acc'] = data_new['mort_acc'][0]

        if data['delinq_2yrs'][0] > 0:
            data_new['delinq_2yrs'] = 0
            current_params_dict['delinq_2yrs'] = data['delinq_2yrs'][0]
            new_params_dict['delinq_2yrs'] = data_new['delinq_2yrs'][0]


        if data['application_type'][0] == 0:
            data_new['application_type'] = 1
            current_params_dict['application_type'] = data['application_type'][0]
            new_params_dict['application_type'] = data_new['application_type'][0]

        #calculating the difference of scores
        # cur_score = np.rint(self.model.predict(data))
        model_input_new = self.format_input(data_new)
        new_score = np.rint(self.model.predict(model_input_new))
        self.score_diff = new_score - cur_score
        # return changes_dict, cur_score, new_score, score_diff, data, data_new
        return current_params_dict, new_params_dict, self.round_dozen(new_score), self.score_diff

# if __name__ == '__main__':
#     # pd.set_option('max_columns', None)
#     #  ------------testing--------------------------------------------
#     feats = ['emp_length', 'annual_inc', 'delinq_2yrs', 'mths_since_last_delinq',
#            'tot_cur_bal', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op',
#            'mort_acc', 'num_actv_bc_tl', 'home_ownership_MORTGAGE',
#            'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
#            'application_type_Individual', 'application_type_Joint App']
#
#     path = 'model/rfr_model.pkl'
#     yscore = Yscore(path)
#
#     samp_list = tst.samp_list
#     best_diff = 0
#     for idx, sample in enumerate(samp_list):
#         sample = np.array([sample])
#         # print(sample1)
#         imp_dict = yscore.feature_weigths(sample, feats)
#         # print(f'{imp_dict=}')
#
#         changes, cur_score, new_score, score_diff, data, new_data = yscore.improve_score(sample, feats)
#
#
#         # diff = math.ceil(new_score/10)*10 - math.ceil(cur_score/10)*10
#         # if diff > best_diff:
#         # if score_diff<= 0:
#         print(data.to_string())
#         print(f'now: {cur_score[0]}, potentially: {new_score[0]}, diff: {score_diff[0]}')
#         print(f'You current score is in range {math.floor((cur_score-0.01)/10)*10} - {math.ceil(cur_score/10)*10}. Your potential score is in range {math.floor(new_score/10)*10} - {math.ceil((new_score+0.01)/10)*10}')
#         print(f'{changes=}')
#         print(idx)
#         print()
#             # best_diff = diff
