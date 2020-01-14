import requests

r = requests.get(
    'https://jsonplaceholder.typicode.com/todos/1')

r_dict = r.json()
print(r.json())
