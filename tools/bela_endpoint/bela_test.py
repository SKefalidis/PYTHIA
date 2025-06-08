import requests

response = requests.post("http://localhost:8001/nerd", json={"question": "Who is the office holder with deputies as Neil Brown and Andrew Peacock?"})
print(response.json())

response = requests.post("http://localhost:8001/nerd", json={"question": "who was richard nixon married to?"})
print(response.json())

response = requests.post("http://localhost:8001/nerd", json={"question": "where was george washington carver from?"})
print(response.json())