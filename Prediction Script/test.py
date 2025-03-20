import socket
import requests

# Force IPv4
socket.setdefaulttimeout(10)

url = "http://127.0.0.1:3000/api/sendText"

headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

data = {
    "chatId": "3113076683@c.us",
    "text": "Anaomly Detected!",
    "session": "default"
}

response = requests.post(url, json=data, headers=headers)

#print("Response Status Code:", response.status_code)
print("Response Text:", response.text)