import numpy as np
import pandas as pd
from shap import TreeExplainer
import pickle
from xgboost import XGBRegressor
import math
import testing as tst


class Yscore:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        with open(self.model_path, 'rb') as f:
            xgb_model: XGBRegressor = pickle.load(f)
        return xgb_model

    def feature_weigths(self, sample: np.ndarray, feat_names: list, top_feat_num=2):
        """
        :param sample: numpy array with user's data
        :param feat_names: np array or list of features
        :param top_feat_num: number of top features you want to see impacting neg and pos the score
        :return: dictionary with top impacting features and their weight
        """
        explainer = TreeExplainer(self.model)
        shap_values = explainer.shap_values(sample)
        feat_imp = pd.DataFrame(np.around(shap_values, 5))
        df_imp = pd.concat((pd.DataFrame(feat_names), feat_imp.T), axis=1)
        df_imp.columns = ['feature', 'importance']
        df_imp.sort_values(by='importance', inplace=True, ascending=False)
        df_imp = pd.concat((df_imp.head(top_feat_num), df_imp.tail(top_feat_num)), axis=0)
        result = dict(zip(df_imp.feature, df_imp.importance))
        return result

    def improve_score(self, sample: np.ndarray, feat_names, feat_num=2):
        changes_dict = {}  # dictionary of updated personal parameters

        data = pd.DataFrame(sample)
        data.columns = feat_names
        data_new = data.copy()

        # changing all features that make sense to be changed and writing to dictionary:
        if data['emp_length'][0] < 10:
            data_new['emp_length'] += 1
            changes_dict['emp_length'] = [data['emp_length'][0], data_new['emp_length'][0]]

        if data['annual_inc'][0] < 6:
            data_new['annual_inc'] += 1
            changes_dict['annual_inc'] = [data['annual_inc'][0], data_new['annual_inc'][0]]

        if (data['mths_since_last_delinq'][0] != -1) & (data['mths_since_last_delinq'][0] < 4):
            data_new['mths_since_last_delinq'] += 1
            changes_dict['mths_since_last_delinq'] = [data['mths_since_last_delinq'][0],
                                                      data_new['mths_since_last_delinq'][0]]

        if data['tot_cur_bal'][0] < 9:
            data_new['tot_cur_bal'] += 1
            changes_dict['tot_cur_bal'] = [data['tot_cur_bal'][0], data_new['tot_cur_bal'][0]]

        if data['mort_acc'][0] > 0:
            data_new['mort_acc'] -= 1
            changes_dict['mort_acc'] = [data['mort_acc'][0], data_new['mort_acc'][0]]

        #calculating the difference of scores
        cur_score = np.rint(self.model.predict(data))
        new_score = np.rint(self.model.predict(data_new))
        score_diff = new_score - cur_score
        # return changes_dict, cur_score, new_score, score_diff, data, data_new
        return changes_dict, cur_score, new_score

# if __name__ == '__main__':
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
#
#     for sample in samp_list:
#         sample = np.array([sample])
#         # print(sample1)
#         imp_dict = yscore.feature_weigths(sample, feats)
#         changes, cur_score, new_score, score_diff, data, new_data = yscore.improve_score(sample, imp_dict, feats)
#
#         if math.ceil(new_score/10)*10 > math.ceil(cur_score/10)*10:
#             print(f'now: {cur_score[0]}, potentially: {new_score[0]}')
#             print(f'You current score is in range {math.floor((cur_score-0.01)/10)*10} - {math.ceil(cur_score/10)*10}. Your potential score is in range {math.floor(new_score/10)*10} - {math.ceil((new_score+0.01)/10)*10}')
#             print(f'{changes=}')
#             print()
