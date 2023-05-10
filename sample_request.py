import requests
import json

data = {"name": "inititial","tags": "example","random_number": 52}

r = requests.post("http://127.0.0.1:8000/statenumbers/", data=json.dumps(data))

r2 = requests.get('http://127.0.0.1:8000/statenumbers/52')

print(r.status_code)
print(r.json())

print(r2.status_code)
print(r2.json())