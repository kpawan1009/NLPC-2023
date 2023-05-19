import requests
from geopy.geocoders import Nominatim
import json
from urllib.request import urlopen

# calling the Nominatim tool
loc = Nominatim(user_agent="GetLoc")
url='http://ipinfo.io/json'
# entering the location name
response=urlopen(url)
data=json.load(response)
print(data)
str=""
url = "https://www.fast2sms.com/dev/bulkV2"
for i in data:
    str=str+(i+" : "+data[i]+" ")
print(str)
message="The Person is in risk ,go and help him , his last location "+str
# print(message)
# sender_id=TXTIND
payload = "message="+message+"&language=english&route=q&numbers=9801777249"

headers = {
'authorization': "tUAB1DLHKTHSr5tu8hhmVnCqkKSLQQleEMOXHm8WmSVKR37fZdTu0KfetqRm",
'Content-Type': "application/x-www-form-urlencoded",
'Cache-Control': "no-cache"
}

response = requests.request("POST", url, data=payload, headers=headers)
print(message)
print(response.text)