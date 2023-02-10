# Import libraries
import json


# Function to read and return json
def read(path: str) -> dict:
    try:
        json_file = open(path)
        json_obj = json.load(json_file)
        return json_obj
    except Exception as e:
        print(e)
