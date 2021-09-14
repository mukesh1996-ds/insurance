import requests

url = 'http://127.0.0.1:5000/results'
r = requests.post(url, json={'age': 19, 'sex': 0, 'bmi': 27.8, 'children': 1, 'smoker': 1, 'region': 1})
print('The price for insurance policy is :', r.json())
