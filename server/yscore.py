import numpy as np
import pandas as pd
from shap import TreeExplainer
import pickle
import xgboost


def feature_weigths(model, sample, feat_names, top_feat_num=2):
    """
    :param model: pickle file with model
    :param sample: numpy array with user's data
    :param feat_names: np array or list of features
    :param top_feat_num: number of top features you want to see impacting neg and pos the score
    :return: dictionary with top impacting features and their weight
    """

    explainer = TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    feat_imp = pd.DataFrame(np.around(shap_values, 5))
    df_imp = pd.concat((pd.DataFrame(feat_names), feat_imp.T), axis=1)
    df_imp.columns = ['feature', 'importance']
    df_imp.sort_values(by='importance', inplace=True, ascending=False)
    df_imp = pd.concat((df_imp.head(top_feat_num), df_imp.tail(top_feat_num)), axis=0)
    result = dict(zip(df_imp.feature, df_imp.importance))
    return result


def improve_score(model, sample, imp_dict, feat_names, feat_num=2):
    # The function is very dirty and needs to be improved
    """
    :param model: pickle file with model
    :param sample: numpy array with user's data
    :param imp_dict: dictionary with top impacting features and their weight (from feature_weigths function)
    :param feat_num: number of top features you want to see impacting neg and pos the score
    :return: sample of improved feature values, updated score (hopefully a higher one), % of score difference (hopefully not negative)
    """
    sample = pd.DataFrame(sample)
    sample.columns=feat_names
    sample_upd = sample.copy()
    cur_score = model.predict(sample)
    new_score = cur_score
    if cur_score <= new_score:
        for i, feat in enumerate(list(imp_dict.keys())):
            if i < feat_num:
                sample_upd[feat] = sample_upd[feat] + 1
            else:
                # if sample_upd[feat] != 0:
                sample_upd[feat] = sample_upd[feat] - 1
        new_score = model.predict(sample_upd)
    score_diff = (new_score - cur_score) / new_score

    return sample_upd, cur_score,  new_score, score_diff

#------------testing--------------------------------------------


sample = np.array([[2, 1, 0, 3, 3, 7, 0, 0, 2, 0, 0, 0, 1, 1, 0]])
feats = ['emp_length', 'annual_inc', 'delinq_2yrs', 'mths_since_last_delinq',
       'tot_cur_bal', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op',
       'mort_acc', 'num_actv_bc_tl', 'home_ownership_MORTGAGE',
       'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
       'application_type_Individual', 'application_type_Joint App']
#
MODEL_PATH = "model/xgbr_model.pkl"
# MODEL_PATH = "model/rfr_model.pkl"
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# print(feature_weigths(model, sample, feats))
# print(model.predict(sample))
imp_dict = feature_weigths(model, sample, feats)
# print(imp_dict)
print(improve_score(model, sample, imp_dict, feats))
# print(model)