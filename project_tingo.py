import requests
import json
API_KEY = "673d20756a04701663d698a716cb107194f60436"

headers = {
    'Content-Type': 'application/json'
}

requestResponse = requests.get("https://api.tiingo.com/tiingo/daily/aapl/prices?startDate=2019-01-02&token=673d20756a04701663d698a716cb107194f60436", headers=headers)
# Save the response JSON to a file
with open('/Users/anshulshirude/JuniorYear/AI/AIProject/response.json', 'w') as file:
    json.dump(requestResponse.json(), file)

# Read the file and print its contents
with open('/Users/anshulshirude/JuniorYear/AI/AIProject/response.json', 'r') as file:
    response_data = json.load(file)
    print(response_data)
print(requestResponse.json())