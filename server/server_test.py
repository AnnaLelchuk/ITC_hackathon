import requests
import json


if __name__ == '__main__':
    BASE_URL = "http://ec2-3-72-250-109.eu-central-1.compute.amazonaws.com:8080/"
    LOCALHOST = "http://127.0.0.1:8080/"

    emp_length = [0, 1, 2, 3, 4, 5, 6]
    annual_inc = [0, 1, 2, 3, 4]
    delinq_2yrs = [0, 1, 2]
    mths_since_last_delinq = [-1, 0, 1, 2],
    tot_cur_bal = [0, 1, 2]
    mo_sin_old_rev_tl_op = [0, 1, 2]
    mo_sin_rcnt_tl = [0, 1, 2]
    mort_acc = [0, 1, 2, 3, 4, 5]
    num_actv_bc_tl = [0, 1, 2]
    home_ownership = [0, 1, 2, 3]
    application_type = [0, 1]

    test_dict = {
            "emp_length": 3,
            "annual_inc": 2,
            "delinq_2yrs": 0,
            "mths_since_last_delinq": -1,
            "tot_cur_bal": 5,
            "mo_sin_old_rev_tl_op": 3,
            "mo_sin_rcnt_rev_tl_op": 7,
            "mort_acc": 4,
            "num_actv_bc_tl": 1,
            "home_ownership": 2,
            "application_type": 0
        }

    test_json = json.dumps(test_dict)


    # test_url = LOCALHOST + 'score?application_type=1&home_ownership=2&' \
    #            'mo_sin_rcnt_rev_tl_op=1&mo_sin_old_rev_tl_op=0&num_actv_bc_tl=0&' \
    #            'annual_inc=1&tot_cur_bal=1&emp_length=2&delinq_2yrs=3&' \
    #            'mths_since_last_delinq=2&mort_acc=2'
    # print(requests.get(LOCALHOST))
    test_url = LOCALHOST + "/score"
    response = requests.get(test_url, params=test_dict)
    print(response.text)