import requests
import json


if __name__ == '__main__':
    BASE_URL = "http://ec2-3-72-250-109.eu-central-1.compute.amazonaws.com:8080/"
    LOCALHOST = "http://127.0.0.1:8080/"

    test_dict = {
        "form": {
            "emp_length": 0,
            "annual_inc": 3,
            "delinq_2yrs": 4,
            "mths_since_last_delinq": -1,
            "tot_cur_bal": 5,
            "mo_sin_old_rev_tl_op": 3,
            "mo_sin_rcnt_tl": 5,
            "mort_acc": 1,
            "num_actv_bc_tl": 3,
            "home_ownership": 2,
            "application_type": 1
        }
    }

    test_json = json.dumps(test_dict)
    test_url = 'http://172.16.0.146:8080/score?application_type=0&home_ownership=1&mo_sin_rcnt_tl=2&' \
               'mo_sin_old_rev_tl_op=5&num_actv_bc_tl=5&annual_inc=2&tot_cur_bal=3&emp_length=8&' \
               'delinq_2yrs=5&mths_since_last_delinq=3&mort_acc=1'
    # print(requests.get(LOCALHOST))
    response = requests.get(test_url)
    print(response.text)