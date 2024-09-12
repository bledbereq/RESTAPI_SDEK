from flask import Flask, request, jsonify
from router import get_category

app = Flask(__name__)

@app.route('/process_json', methods=['POST'])
def process_json():
    try:
        json_data = request.get_json()

        if not json_data:
            return jsonify({"error": "No JSON provided"}), 400
        
        result = []
        for sheet_name, records in json_data.items():
            for record in records:
                record_id = record.get('id', len(result) + 1)
                name = record.get('name', '')
                content = record.get('content', '')
                description = record.get('description', '')
                name_en = record.get('name_en', '')
                images = record.get('images', '')
                
                result.append(get_category(id=record_id,
                                           name=name, 
                                           content=content,
                                           description=description,
                                           name_en=name_en,
                                           images=images))

        return jsonify({"processed_data": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)