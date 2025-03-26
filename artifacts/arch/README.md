# CSV Validator

This project is a web application for validating CSV files against a set of rules using Google's Gemini API. It allows users to upload a CSV file and a rules file, validate the data, correct errors, and download the updated file.

## Features

- **Upload CSV and Rules Files**: Users can upload a CSV file containing data and a plain text file with validation rules.
- **Data Validation**: The application uses Google's Gemini API to parse the rules and validate the data in the CSV file.
- **Interactive Error Correction**: Users can view validation errors and correct them directly in the web interface.
- **Download Corrected CSV**: After making corrections, users can download the updated CSV file.

## How It Works

1. **File Upload**: Users upload a CSV file and a rules file. The CSV file contains the data to be validated, and the rules file contains plain English validation rules.

2. **Rule Parsing**: The application uses Google's Gemini API to convert the plain English rules into a structured JSON format that can be used for validation.

3. **Data Validation**: The application validates each row in the CSV file against the parsed rules. It checks for various conditions such as amount matching, balance validation, currency code validation, and date checks.

4. **Error Display and Correction**: Validation errors are displayed in the UI, where users can correct the data points with errors. The UI provides input fields for easy correction.

5. **Download Updated File**: Once corrections are made, users can download the corrected CSV file.

## Future Scope

- **Enhanced Rule Parsing**: Improve the rule parsing capabilities to support more complex validation scenarios and extract rules for complex files.


## Prerequisites

- Python 3.7 or higher
- Google Gemini API key

## Installation

1. **Clone the repository:**

   Clone the repository from Github and open command terminal

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Google Gemini API key:**

   Replace the placeholder API key in `app.py` with your actual Google Gemini API key:
   ```python
   client = genai.Client(api_key="your-gemini-api-key")
   ```

## Usage

1. **Run the application:**

   ```bash
   python app.py
   ```

2. **Open your web browser and go to:**

   ```
   http://127.0.0.1:5000
   ```

3. **Upload your CSV and rules files:**

   - The CSV file should contain the data you want to validate.
   - The rules file should be a plain text file containing the validation rules in English.

4. **View and correct errors:**

   - The application will display a summary of validation errors.
   - You can correct errors directly in the web interface.

5. **Download the corrected CSV file:**

   - Once corrections are made, download the corrected file for further use.

## File Structure

- `app.py`: Main application file containing the Flask app and validation logic.
- `templates/index.html`: HTML template for the web interface.
- `requirements.txt`: List of Python dependencies.

## Dependencies

- Flask
- pandas
- google-genai
- iso4217parse
