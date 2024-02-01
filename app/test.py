import requests
import json

def test_api_connection():
    url = "http://127.0.0.1:5000/v1/completions"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "prompt": "why is the sky blue?.\n\n",
        "max_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raises an error for HTTP codes 400 or higher
        completion = response.json().get('choices', [{}])[0].get('text', '')
        print("API Response:", completion)
    except requests.RequestException as e:
        print("Error connecting to the API:", e)

# Running the test function
test_api_connection()
