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
import PyPDF2
import csv

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
    
    # Validate rules file extension
    allowed_extensions = {'.txt', '.csv', '.pdf'}
    rules_ext = os.path.splitext(rules_file.filename)[1].lower()
    if rules_ext not in allowed_extensions:
        return jsonify({'error': 'Rules file must be .txt, .csv, or .pdf'}), 400
    
    try:
        # Save uploaded files
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(csv_file.filename))
        rules_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(rules_file.filename))
        
        csv_file.save(csv_path)
        rules_file.save(rules_path)
        
        # Process files
        df = pd.read_csv(csv_path)
        
        # Extract rules from file
        raw_rules_text = extract_rules_from_file(rules_path)
        
        # Convert to plain English format
        normalized_rules = normalize_rules_to_english(raw_rules_text, list(df.columns))
        
        # Parse rules into structured format
        structured_rules = parse_rules_with_llm(normalized_rules, list(df.columns))
        
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
        
        # Save validation results for later use
        session_data = {
            'csv_path': csv_path,
            'validation_results': {str(k): v for k, v in validation_results.items()}
        }
        
        temp_session_file = os.path.join(app.config['UPLOAD_FOLDER'], 'session.json')
        with open(temp_session_file, 'w') as f:
            json.dump(session_data, f)
        
        return jsonify({
            'preview': preview
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def extract_rules_from_file(file_path: str) -> str:
    """Extract rules text from different file formats"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        elif file_extension == '.csv':
            rules_text = []
            with open(file_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                headers = next(csv_reader, None)  # Skip header row if exists
                for row in csv_reader:
                    if row:  # Skip empty rows
                        rules_text.append(' '.join(str(cell) for cell in row if cell))
            return '\n'.join(rules_text)
            
        elif file_extension == '.pdf':
            rules_text = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    rules_text.append(page.extract_text())
            return '\n'.join(rules_text)
            
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
    except Exception as e:
        raise Exception(f"Error reading rules file: {str(e)}")

def normalize_rules_to_english(rules_text: str, columns: List[str]) -> str:
    """Convert rules to plain English format using Gemini"""
    prompt = f"""
    Convert the following data validation rules into clear, plain English statements.
    The data has these columns: {', '.join(columns)}
    
    Original rules:
    {rules_text}
    
    Convert each rule into a clear statement following this format:
    - Rule Type: [type of validation]
    - Columns: [affected columns]
    - Condition: [what needs to be validated]
    - Error Message: [what to show if validation fails]
    
    Example:
    - Rule Type: amount_match
    - Columns: Transaction_Amount, Reported_Amount
    - Condition: Values should match within 1% tolerance for cross-currency transactions
    - Error Message: Transaction amount mismatch exceeds permitted tolerance
    
    Please convert ALL rules to this format, maintaining their business logic.
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text
        
    except Exception as e:
        raise Exception(f"Error normalizing rules: {str(e)}")

def parse_rules_with_llm(rules_text: str, columns: List[str]) -> List[Dict]:
    """Parse normalized rules into structured JSON format"""
    prompt = f"""
    Convert these plain English validation rules into a structured JSON format.
    Available columns: {', '.join(columns)}

    You MUST use ONLY the following rule types:
    1. "amount_match" - For comparing monetary amounts
       Parameters: {{"tolerance_percentage": float, "cross_currency_flag_column": string}}
    
    2. "balance" - For account balance validation
       Parameters: {{"min_balance": float, "account_type_column": string}}
    
    3. "currency" - For currency code and cross-border validations
       Parameters: {{"cross_border_limit": float, "documentation_threshold": float}}
    
    4. "country" - For country-specific validations
       Parameters: {{"accepted_countries": [string], "high_risk_countries": [string]}}
    
    5. "date" - For date-related validations
       Parameters: {{"max_age_days": int, "max_value_date_diff_days": int, "exclude_weekends": boolean}}
    
    6. "required_field" - For mandatory field validation
       Parameters: {{}} (no additional parameters needed)
    
    7. "numeric_range" - For number range validation
       Parameters: {{"min": float, "max": float}}
    
    8. "format" - For pattern/regex validation
       Parameters: {{"pattern": string}}
    
    9. "dependency" - For related field validation
       Parameters: {{"condition": "required"|"matching"}}
    
    10. "unique" - For uniqueness validation
       Parameters: {{}} (no additional parameters needed)

    Rules to convert:
    {rules_text}

    Convert to this JSON structure:
    [
        {{
            "type": "one_of_above_types",
            "columns": ["affected_columns"],
            "condition": "validation_condition",
            "parameters": {{ "param_key": "param_value" }},
            "error_message": "clear_error_message"
        }}
    ]

    Important:
    - Use ONLY the rule types listed above
    - Include appropriate parameters for each rule type
    - Make error messages clear and specific
    - Ensure column names match the available columns
    - Return ONLY the JSON array, no explanation

    Return ONLY the JSON array, no explanation.
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        # Extract and parse JSON from response
        content = response.text
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            rules = json.loads(json_match.group(0))
        else:
            rules = json.loads(content.strip())
        
        # Validate that all rules use supported types
        supported_types = {
            "amount_match", "balance", "currency", "country", "date",
            "required_field", "numeric_range", "format", "dependency", "unique"
        }
        
        for rule in rules:
            if rule.get('type', '').lower() not in supported_types:
                raise ValueError(f"Unsupported rule type: {rule.get('type')}")
        
        return rules
            
    except Exception as e:
        raise Exception(f"Error parsing rules: {str(e)}")

def validate_data(df: pd.DataFrame, rules: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, str]]]:
    """
    Enhanced validation function with descriptive error messages
    """
    validation_results = {}
    print("\n=== Starting Validation ===")
    
    # Preprocess date columns and other data types
    date_columns = ['Transaction_Date', 'Value_Date', 'Settlement_Date']  # Add all possible date columns
    numeric_columns = ['Transaction_Amount', 'Reported_Amount', 'Account_Balance', 'Exchange_Rate']
    
    # Data type preprocessing
    for col in df.columns:
        if col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        elif col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for index, row in df.iterrows():
        print(f"\nValidating Row {index + 1}:")
        row_errors = []
        
        for rule in rules:
            try:
                rule_type = rule.get('type', '').lower()
                columns = rule.get('columns', [])
                condition = rule.get('condition', '')
                parameters = rule.get('parameters', {})
                
                # Create descriptive error message based on rule type
                error_description = get_rule_description(rule_type, condition, parameters)
                correction_guidance = get_correction_guidance(rule_type, parameters)
                
                print(f"\n  Checking {rule_type} rule for columns: {columns}")
                
                # Validate required columns exist
                if not all(col in df.columns for col in columns):
                    missing_cols = [col for col in columns if col not in df.columns]
                    print(f"  ⚠️ Missing columns: {missing_cols}")
                    continue
                
                valid = True
                additional_info = ""
                
                # Enhanced validation based on rule type
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
                    
                elif rule_type == "required_field":
                    valid, additional_info = validate_required_field(row, columns)
                    
                elif rule_type == "numeric_range":
                    valid, additional_info = validate_numeric_range(row, columns[0], parameters)
                    
                elif rule_type == "format":
                    valid, additional_info = validate_format(row, columns[0], parameters)
                    
                elif rule_type == "dependency":
                    valid, additional_info = validate_dependency(row, columns, parameters)
                    
                elif rule_type == "unique":
                    valid, additional_info = validate_unique(df, row, columns[0], index)
                
                if not valid:
                    error = {
                        'columns': ', '.join(columns),
                        'values': {col: str(row[col]) for col in columns if col in df.columns},
                        'rule_description': error_description,
                        'correction_guidance': correction_guidance,
                        'validation_details': additional_info
                    }
                    row_errors.append(error)
                    print(f"  ❌ Rule violated: {error_description}")
                    print(f"     Correction needed: {correction_guidance}")
            
            except Exception as e:
                print(f"  ⚠️ Error processing rule: {str(e)}")
                continue
        
        if row_errors:
            validation_results[index] = row_errors
            print(f"  Total errors for row {index + 1}: {len(row_errors)}")
        else:
            print(f"  ✅ Row {index + 1} passed all validations")
    
    return validation_results

def validate_amount_match(row: pd.Series, parameters: Dict) -> tuple[bool, str]:
    """Enhanced amount matching validation"""
    try:
        trans_amt = float(row['Transaction_Amount'])
        rep_amt = float(row['Reported_Amount'])
        tolerance = float(parameters.get('tolerance_percentage', 1.0)) / 100
        is_cross_currency = bool(row.get(parameters.get('cross_currency_flag_column', 'Is_Cross_Currency'), False))
        
        # Handle zero amounts
        if trans_amt == 0 and rep_amt == 0:
            return True, ""
        
        # Avoid division by zero
        max_amount = max(abs(trans_amt), abs(rep_amt))
        if max_amount == 0:
            return False, "Invalid amount: zero value detected"
        
        difference_percentage = abs(trans_amt - rep_amt) / max_amount
        
        if is_cross_currency:
            # Check if exchange rate is provided
            if 'Exchange_Rate' in row:
                exchange_rate = float(row['Exchange_Rate'])
                adjusted_amount = trans_amt * exchange_rate
                difference_percentage = abs(adjusted_amount - rep_amt) / max(abs(adjusted_amount), abs(rep_amt))
        
        valid = difference_percentage <= tolerance
        info = f"Difference: {difference_percentage:.2%} (tolerance: {tolerance:.2%})" if not valid else ""
        
        return valid, info
    except Exception as e:
        return False, f"Amount validation error: {str(e)}"

def validate_required_field(row: pd.Series, columns: List[str]) -> tuple[bool, str]:
    """Validate required fields are not empty"""
    try:
        for column in columns:
            value = row[column]
            if pd.isna(value) or str(value).strip() == "":
                return False, f"Required field '{column}' is empty"
        return True, ""
    except Exception as e:
        return False, f"Required field validation error: {str(e)}"

def validate_numeric_range(row: pd.Series, column: str, parameters: Dict) -> tuple[bool, str]:
    """Validate numeric value falls within specified range"""
    try:
        value = float(row[column])
        min_value = float(parameters.get('min', float('-inf')))
        max_value = float(parameters.get('max', float('inf')))
        
        if value < min_value:
            return False, f"Value {value} is below minimum {min_value}"
        if value > max_value:
            return False, f"Value {value} exceeds maximum {max_value}"
        
        return True, ""
    except Exception as e:
        return False, f"Numeric range validation error: {str(e)}"

def validate_format(row: pd.Series, column: str, parameters: Dict) -> tuple[bool, str]:
    """Validate field format using regex patterns"""
    try:
        value = str(row[column])
        pattern = parameters.get('pattern', '')
        if not pattern:
            return True, ""
        
        if not re.match(pattern, value):
            return False, f"Value '{value}' does not match required format"
        
        return True, ""
    except Exception as e:
        return False, f"Format validation error: {str(e)}"

def validate_dependency(row: pd.Series, columns: List[str], parameters: Dict) -> tuple[bool, str]:
    """Validate dependent field relationships"""
    try:
        main_field = columns[0]
        dependent_field = columns[1]
        condition = parameters.get('condition', 'required')
        
        main_value = row[main_field]
        dependent_value = row[dependent_field]
        
        if condition == 'required':
            if not pd.isna(main_value) and pd.isna(dependent_value):
                return False, f"'{dependent_field}' is required when '{main_field}' is present"
        elif condition == 'matching':
            if not pd.isna(main_value) and not pd.isna(dependent_value):
                if str(main_value) != str(dependent_value):
                    return False, f"'{dependent_field}' must match '{main_field}'"
        
        return True, ""
    except Exception as e:
        return False, f"Dependency validation error: {str(e)}"

def validate_unique(df: pd.DataFrame, row: pd.Series, column: str, current_index: int) -> tuple[bool, str]:
    """Validate field uniqueness across the dataset"""
    try:
        value = row[column]
        if pd.isna(value):
            return True, ""
        
        # Check for duplicates excluding current row
        duplicate_mask = (df[column] == value) & (df.index != current_index)
        if duplicate_mask.any():
            duplicate_indices = df.index[duplicate_mask].tolist()
            return False, f"Duplicate value found in rows: {[i+1 for i in duplicate_indices]}"
        
        return True, ""
    except Exception as e:
        return False, f"Uniqueness validation error: {str(e)}"

def validate_currency(row: pd.Series, parameters: Dict) -> tuple[bool, str]:
    """Enhanced currency validation"""
    try:
        currency = str(row['Currency'])
        amount = float(row['Transaction_Amount'])
        
        # Validate ISO 4217 currency code
        try:
            currency_info = iso4217parse.parse(currency)
            if not currency_info:
                return False, f"Invalid currency code: {currency}"
        except ValueError:
            return False, f"Invalid currency code: {currency}"
        
        # Check cross-border transaction limits
        is_cross_border = bool(row.get('Is_Cross_Border', False))
        if is_cross_border:
            limit = float(parameters.get('cross_border_limit', float('inf')))
            if amount > limit:
                return False, f"Amount {amount} exceeds cross-border limit of {limit}"
            
            # Check for required documentation
            if amount > float(parameters.get('documentation_threshold', 10000)):
                if not row.get('Documentation_Reference'):
                    return False, "Missing documentation reference for large cross-border transaction"
        
        # Validate currency pairs for cross-currency transactions
        if row.get('Counter_Currency'):
            counter_currency = str(row['Counter_Currency'])
            try:
                counter_currency_info = iso4217parse.parse(counter_currency)
                if not counter_currency_info:
                    return False, f"Invalid counter currency code: {counter_currency}"
            except ValueError:
                return False, f"Invalid counter currency code: {counter_currency}"
        
        return True, ""
    except Exception as e:
        return False, f"Currency validation error: {str(e)}"

def validate_date(row: pd.Series, parameters: Dict) -> tuple[bool, str]:
    """Enhanced date validation"""
    try:
        trans_date = row['Transaction_Date']
        if not isinstance(trans_date, pd.Timestamp):
            return False, "Invalid date format"
        
        today = pd.Timestamp.now().normalize()
        
        # Check future date
        if trans_date > today:
            return False, "Transaction date is in the future"
        
        # Check old transactions
        max_age_days = int(parameters.get('max_age_days', 365))
        days_old = (today - trans_date).days
        if days_old > max_age_days:
            return False, f"Transaction is {days_old} days old (max allowed: {max_age_days})"
        
        # Check value date if present
        if 'Value_Date' in row:
            value_date = row['Value_Date']
            if isinstance(value_date, pd.Timestamp):
                max_value_date_diff = int(parameters.get('max_value_date_diff_days', 5))
                date_diff = abs((value_date - trans_date).days)
                if date_diff > max_value_date_diff:
                    return False, f"Value date differs from transaction date by {date_diff} days (max allowed: {max_value_date_diff})"
        
        # Check for weekend/holiday processing if specified
        if parameters.get('exclude_weekends', False):
            if trans_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                return False, "Transaction date falls on weekend"
        
        return True, ""
    except Exception as e:
        return False, f"Date validation error: {str(e)}"

def get_rule_description(rule_type: str, condition: str, parameters: Dict) -> str:
    """Generate descriptive explanation of the validation rule"""
    descriptions = {
        "amount_match": (
            f"Amounts must match within {parameters.get('tolerance_percentage', 1)}% tolerance"
            f"{' for cross-currency transactions' if parameters.get('cross_currency_flag_column') else ''}"
        ),
        "balance": (
            f"Account balance must be at least {parameters.get('min_balance', 0)} "
            f"for {parameters.get('account_type_column', 'all')} accounts"
        ),
        "currency": (
            f"Currency must be valid ISO code. "
            f"Cross-border transactions limited to {parameters.get('cross_border_limit', 'unlimited')}. "
            f"Documentation required above {parameters.get('documentation_threshold', 10000)}"
        ),
        "country": (
            f"Country must be in accepted list: {parameters.get('accepted_countries', [])}. "
            f"Additional checks for high-risk countries: {parameters.get('high_risk_countries', [])}"
        ),
        "date": (
            f"Date must be valid, not future, within {parameters.get('max_age_days', 365)} days old"
            f"{', excluding weekends' if parameters.get('exclude_weekends') else ''}"
        ),
        "required_field": "Field is mandatory and cannot be empty",
        "numeric_range": (
            f"Value must be between {parameters.get('min', '-∞')} and {parameters.get('max', '∞')}"
        ),
        "format": f"Value must match pattern: {parameters.get('pattern', 'any')}",
        "dependency": (
            "Fields must satisfy dependency condition: "
            f"{parameters.get('condition', 'required')}"
        ),
        "unique": "Value must be unique across all records"
    }
    return descriptions.get(rule_type, "Unknown validation rule")

def get_correction_guidance(rule_type: str, parameters: Dict) -> str:
    """Generate guidance on how to correct the validation error"""
    guidance = {
        "amount_match": (
            "Ensure the amounts match within the specified tolerance. "
            "For cross-currency transactions, verify the exchange rate is correct."
        ),
        "balance": (
            "Adjust the account balance to meet the minimum requirement "
            "for the specified account type."
        ),
        "currency": (
            "Use valid ISO currency code. For cross-border transactions, "
            "ensure proper documentation and stay within limits."
        ),
        "country": (
            "Use a country from the accepted list. For high-risk countries, "
            "ensure additional documentation is provided."
        ),
        "date": (
            "Enter a valid date that is not in the future and within the allowed age range. "
            "For value dates, ensure they are within permitted difference from transaction date."
        ),
        "required_field": (
            "Provide a non-empty value for this mandatory field."
        ),
        "numeric_range": (
            f"Enter a number between {parameters.get('min', '-∞')} and {parameters.get('max', '∞')}."
        ),
        "format": (
            "Enter the value in the correct format according to the specified pattern."
        ),
        "dependency": (
            "Ensure related fields are properly filled according to the dependency rules."
        ),
        "unique": (
            "Provide a unique value that is not used in any other record."
        )
    }
    return guidance.get(rule_type, "Please review and correct the data according to validation rules.")

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