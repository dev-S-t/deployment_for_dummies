import requests

url = "http://127.0.0.1:8000/process_text"
data = {"text": "having cold and fever"}

response = requests.post(url, json=data)
print(response.json())
