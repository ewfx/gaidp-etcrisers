from flask import Blueprint, request, jsonify
from src.controllers import read_file_content, read_raw_text

routes = Blueprint('routes', __name__)

@routes.route('/read-file', methods=['POST'])
def read_file():
    file_path = request.json.get('file_path')
    if not file_path:
        return jsonify({'error': 'file_path is required'}), 400
    content = read_file_content(file_path)
    return jsonify({'content': content})

@routes.route('/read-text', methods=['POST'])
def read_text():
    raw_text = request.json.get('text')
    if raw_text is None:
        return jsonify({'error': 'text is required'}), 400
    return jsonify({'content': read_raw_text(raw_text)})