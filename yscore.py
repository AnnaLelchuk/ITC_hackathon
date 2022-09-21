import numpy as np
import pandas as pd
import shap


def feature_weigths(model, sample, feat_names, top_feat_num=2):
    """
    :param model: pickle file with model
    :param sample: numpy array with user's data
    :param feat_names: np array or list of features
    :param top_feat_num: number of top features you want to see impacting neg and pos the score
    :return: dictionary with top impacting features and their weight
    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    feat_imp = pd.DataFrame(np.around(shap_values, 5))
    df_imp = pd.concat((pd.DataFrame(feat_names), feat_imp), axis=1)
    df_imp.columns = ['feature', 'importance']
    df_imp.sort_values(by='importance', inplace=True, ascending=False)
    df_imp = pd.concat((df_imp.head(top_feat_num), df_imp.tail(top_feat_num)), axis=0)

    # dictionary with top-2 pos. and neg. affecting features
    result = dict(zip(df_imp.feature, df_imp.importance))
    return result


def improve_score(model, sample, imp_dict, feat_num=2):
    # The function is very dirty and needs to be improved
    """
    :param model: pickle file with model
    :param sample: numpy array with user's data
    :param imp_dict: dictionary with top impacting features and their weight (from feature_weigths function)
    :param feat_num: number of top features you want to see impacting neg and pos the score
    :return: sample of improved feature values, updated score (hopefully a higher one), % of score difference (hopefully not negative)
    """

    sample_upd = sample.copy()
    cur_score = model.predict(sample)
    new_score = cur_score
    if cur_score == new_score:
        for i, feat in enumerate(list(imp_dict.keys())):
            if i < feat_num:
                sample_upd[feat] = sample_upd[feat] + 1
            else:
                sample_upd[feat] = sample_upd[feat] - 1
        new_score = model.predict(sample_upd)
    score_diff = (new_score - cur_score) / new_score

    return sample_upd, new_score, score_diff
