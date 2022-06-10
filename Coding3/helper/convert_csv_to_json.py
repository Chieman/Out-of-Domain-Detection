import csv
import json
csv_file_path = "//home/an/Documents/out-of-domain/Coding3/WSUntitled spreadsheet - Sheet1.csv"
data_dict = {}
json_file_path = "/home/an/Documents/out-of-domain/Coding3/WSUntitled spreadsheet - Sheet1.json"
with open(csv_file_path, encoding='utf-8') as csv_file_handler:
    csv_reader = csv.DictReader(csv_file_handler)
    key = "train"
    data_dict[key] = []
    for rows in csv_reader:
        data_dict[key].append([rows['text'], rows['labels']])
with open(json_file_path, 'w') as json_file_handler:
    json_file_handler.write(json.dumps(data_dict, indent=3, ensure_ascii=False))