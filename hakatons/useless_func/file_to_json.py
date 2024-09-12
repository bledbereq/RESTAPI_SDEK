import pandas as pd

# Define the path to the uploaded Excel file


excel_file = 'west.xlsx'
print("Start programm")
# Load the Excel file
data = pd.read_excel(excel_file, sheet_name=None)
print("Reading")
# Convert the data to JSON format
json_data = {sheet: df.to_dict(orient='records') for sheet, df in data.items()}

# Save the JSON data to a file
output_file = './output_data.json'
with open(output_file, 'w', encoding='utf-8') as json_file:
    import json
    json.dump(json_data, json_file, ensure_ascii=False, indent=4)

output_file
