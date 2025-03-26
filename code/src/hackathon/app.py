from flask import Flask, request, render_template, jsonify, send_file
import pandas as pd
import json
import os
import tempfile
from werkzeug.utils import secure_filename
from google import genai
from datetime import datetime, timedelta
import re
from typing import Dict, List, Any
import iso4217parse  # For currency code validation

app = Flask(__name__)

# Configure API key - in production, use environment variables
# Google provides free credits for Gemini API
client = genai.Client(api_key="AIzaSyCufTlSlTUnx81XekgXMfrRcAtULUGoMus")  # Replace with your Gemini API key

UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'csv_file' not in request.files or 'rules_file' not in request.files:
        return jsonify({'error': 'Both CSV and rules files are required'}), 400
    
    csv_file = request.files['csv_file']
    rules_file = request.files['rules_file']
    
    if csv_file.filename == '' or rules_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(csv_file.filename))
    rules_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(rules_file.filename))
    
    csv_file.save(csv_path)
    rules_file.save(rules_path)
    
    # Process files
    try:
        df = pd.read_csv(csv_path)
        with open(rules_path, 'r') as f:
            rules_text = f.read()
        
        # Parse rules using Google's Gemini API
        structured_rules = parse_rules_with_llm(rules_text, list(df.columns))
        
        # Validate data
        validation_results = validate_data(df, structured_rules)
        
        # Generate preview data
        preview = {
            'columns': list(df.columns),
            'sample_data': df.head(5).to_dict('records'),
            'total_rows': len(df),
            'rules': structured_rules,
            'validation_summary': {
                'total_errors': sum(len(errors) for errors in validation_results.values()),
                'rows_with_errors': len([row for row, errors in validation_results.items() if errors])
            }
        }
        
        # Save validation results and file paths for later retrieval
        session_data = {
            'csv_path': csv_path,
            'validation_results': validation_results
        }
        
        temp_session_file = os.path.join(app.config['UPLOAD_FOLDER'], 'session.json')
        with open(temp_session_file, 'w') as f:
            # Convert keys to strings since JSON requires string keys
            serializable_results = {str(k): v for k, v in validation_results.items()}
            json.dump({'csv_path': csv_path, 'validation_results': serializable_results}, f)
        
        return jsonify({
            'preview': preview,
            'validation_results': {str(k): v for k, v in validation_results.items()}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def parse_rules_with_llm(rules_text, columns):
    """
    Enhanced rule parsing to support complex business rules
    """
    try:
        prompt = f"""
        Convert the following complex banking validation rules into a structured JSON format.
        The CSV file has these columns: {', '.join(columns)}
        
        Rules:
        {rules_text}
        
        Create a JSON array where each rule is an object with:
        1. "type": The type of validation ("amount_match", "balance", "currency", "country", "date", etc.)
        2. "column": The main column(s) the rule applies to
        3. "condition": The validation condition
        4. "parameters": Additional parameters needed for validation
        5. "error_message": Clear error message
        
        Example format:
        [
            {{
                "type": "amount_match",
                "columns": ["Transaction_Amount", "Reported_Amount"],
                "condition": "match_with_tolerance",
                "parameters": {{
                    "tolerance_percentage": 1.0,
                    "cross_currency_flag_column": "Is_Cross_Currency"
                }},
                "error_message": "Transaction amount mismatch exceeds permitted tolerance"
            }}
        ]
        
        Return ONLY the JSON with no explanation.
        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        
        # Extract and parse JSON from response
        content = response.text
        import re
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            structured_rules = json.loads(json_match.group(0))
        else:
            structured_rules = json.loads(content.strip())
            
        return structured_rules
    
    except Exception as e:
        print(f"Error parsing rules: {e}")
        return []

def validate_data(df: pd.DataFrame, rules: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, str]]]:
    """
    Enhanced validation function supporting complex banking rules
    """
    validation_results = {}
    print("\n=== Starting Validation ===")
    
    # Preprocess date columns
    date_columns = ['Transaction_Date']  # Add other date columns as needed
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    for index, row in df.iterrows():
        print(f"\nValidating Row {index + 1}:")
        row_errors = []
        
        for rule in rules:
            rule_type = rule.get('type')
            columns = rule.get('columns', [])
            condition = rule.get('condition')
            parameters = rule.get('parameters', {})
            error_message = rule.get('error_message', 'Validation error')  # Default message if not provided
            
            print(f"\n  Checking {rule_type} rule for columns: {columns}")
            
            # Validate required columns exist
            if not all(col in df.columns for col in columns):
                missing_cols = [col for col in columns if col not in df.columns]
                print(f"  ⚠️ Missing columns: {missing_cols}")
                continue
            
            valid = True
            additional_info = ""
            
            # Apply appropriate validation based on rule type
            if rule_type == "amount_match":
                valid, additional_info = validate_amount_match(row, parameters)
            elif rule_type == "balance":
                valid, additional_info = validate_balance(row, parameters)
            elif rule_type == "currency":
                valid, additional_info = validate_currency(row, parameters)
            elif rule_type == "country":
                valid, additional_info = validate_country(row, parameters)
            elif rule_type == "date":
                valid, additional_info = validate_date(row, parameters)
            
            if not valid:
                error = {
                    'columns': ', '.join(columns),  # Join columns into a string
                    'values': {col: str(row[col]) for col in columns if col in df.columns},
                    'error': f"{error_message}. {additional_info}".strip()
                }
                row_errors.append(error)
                print(f"  ❌ Adding error: {error['error']}")
        
        if row_errors:
            validation_results[index] = row_errors
            print(f"  Total errors for row {index + 1}: {len(row_errors)}")
        else:
            print(f"  ✅ Row {index + 1} passed all validations")
    
    print("\n=== Validation Complete ===")
    print(f"Total rows with errors: {len(validation_results)}")
    return validation_results

def validate_amount_match(row: pd.Series, parameters: Dict) -> tuple[bool, str]:
    """Validates transaction amount matching with tolerance for cross-currency"""
    try:
        trans_amt = float(row['Transaction_Amount'])
        rep_amt = float(row['Reported_Amount'])
        tolerance = parameters.get('tolerance_percentage', 1.0) / 100
        is_cross_currency = row.get(parameters.get('cross_currency_flag_column', 'Is_Cross_Currency'), False)
        
        if is_cross_currency:
            # Allow tolerance for cross-currency transactions
            difference_percentage = abs(trans_amt - rep_amt) / max(abs(trans_amt), abs(rep_amt))
            valid = difference_percentage <= tolerance
            info = f"Difference: {difference_percentage:.2%}" if not valid else ""
        else:
            # Exact match required for same currency
            valid = abs(trans_amt - rep_amt) < 0.01  # Account for floating point precision
            info = f"Difference: {abs(trans_amt - rep_amt):.2f}" if not valid else ""
        
        return valid, info
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def validate_balance(row: pd.Series, parameters: Dict) -> tuple[bool, str]:
    """Validates account balance with overdraft consideration"""
    try:
        balance = float(row['Account_Balance'])
        is_od_account = str(row.get('Account_Type', '')).upper() == 'OD'
        
        if balance < 0 and not is_od_account:
            return False, f"Negative balance ({balance:.2f}) in non-overdraft account"
        return True, ""
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def validate_currency(row: pd.Series, parameters: Dict) -> tuple[bool, str]:
    """Validates currency codes and cross-border transaction limits"""
    try:
        currency = str(row['Currency'])
        amount = float(row['Transaction_Amount'])
        
        # Validate ISO 4217 currency code
        try:
            iso4217parse.parse(currency)
            currency_valid = True
        except ValueError:
            return False, f"Invalid currency code: {currency}"
        
        # Check cross-border transaction limits
        if row.get('Is_Cross_Border', False):
            limit = float(parameters.get('cross_border_limit', float('inf')))
            if amount > limit:
                return False, f"Amount {amount} exceeds cross-border limit of {limit}"
        
        return True, ""
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def validate_country(row: pd.Series, parameters: Dict) -> tuple[bool, str]:
    """Validates country and cross-border transaction requirements"""
    try:
        country = str(row['Country'])
        amount = float(row['Transaction_Amount'])
        accepted_countries = parameters.get('accepted_countries', [])
        
        if accepted_countries and country not in accepted_countries:
            return False, f"Country {country} not in accepted list"
        
        # Check for mandatory remarks on large cross-border transactions
        if amount > 10000 and row.get('Is_Cross_Border', False):
            if not row.get('Transaction_Remarks'):
                return False, "Missing mandatory remarks for large cross-border transaction"
        
        return True, ""
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def validate_date(row: pd.Series, parameters: Dict) -> tuple[bool, str]:
    """Validates transaction dates"""
    try:
        trans_date = row['Transaction_Date']
        if not isinstance(trans_date, pd.Timestamp):
            return False, "Invalid date format"
        
        today = pd.Timestamp.now().normalize()
        
        # Check future date
        if trans_date > today:
            return False, "Transaction date is in the future"
        
        # Check old transactions
        days_old = (today - trans_date).days
        if days_old > 365:
            return False, f"Transaction is {days_old} days old"
        
        return True, ""
    except Exception as e:
        return False, f"Validation error: {str(e)}"

@app.route('/get_errors', methods=['GET'])
def get_errors():
    try:
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'session.json'), 'r') as f:
            session_data = json.load(f)
        
        csv_path = session_data['csv_path']
        validation_results = session_data['validation_results']
        
        df = pd.read_csv(csv_path)
        
        # Convert row indices from string back to int
        error_rows = {}
        for str_idx, errors in validation_results.items():
            idx = int(str_idx)
            if errors:  # Only include rows with errors
                row_data = df.iloc[idx].to_dict()
                error_rows[str_idx] = {
                    'data': row_data,
                    'errors': errors
                }
        
        return jsonify({'error_rows': error_rows})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export_corrected', methods=['POST'])
def export_corrected():
    try:
        corrections = request.json.get('corrections', {})
        
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'session.json'), 'r') as f:
            session_data = json.load(f)
        
        csv_path = session_data['csv_path']
        df = pd.read_csv(csv_path)
        
        # Apply corrections
        for str_idx, row_corrections in corrections.items():
            idx = int(str_idx)
            for column, new_value in row_corrections.items():
                df.at[idx, column] = new_value
        
        # Save corrected file
        corrected_path = os.path.join(app.config['UPLOAD_FOLDER'], 'corrected.csv')
        df.to_csv(corrected_path, index=False)
        
        return jsonify({'success': True, 'message': 'File corrected and ready for download'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_corrected')
def download_corrected():
    corrected_path = os.path.join(app.config['UPLOAD_FOLDER'], 'corrected.csv')
    if os.path.exists(corrected_path):
        return send_file(corrected_path, as_attachment=True, download_name='corrected.csv')
    else:
        return jsonify({'error': 'Corrected file not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)