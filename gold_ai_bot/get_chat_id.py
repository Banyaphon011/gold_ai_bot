import requests

TOKEN = "8797930677:AAEDwZp12jr3Vj2qn8yyRyFVYslEfMe1zuI"

url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"

print(requests.get(url).text)