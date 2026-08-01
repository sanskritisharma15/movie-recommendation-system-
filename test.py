import requests

session = requests.Session()
session.trust_env = False  # ignore any proxy env vars

resp = session.get(
    "http://www.omdbapi.com/",
    params={"t": "The Notebook", "apikey": "ddd0246e"},
    timeout=5
)
print(resp.status_code)
print(resp.text[:300])