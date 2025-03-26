from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
import docx
import pandas as pd
import os
import boto3
from botocore.exceptions import NoCredentialsError
# from openai import OpenAI
from google import genai
import json

from dotenv import load_dotenv
load_dotenv()

client_genai = genai.Client(
    api_key = os.environ['GENAI_API_KEY']
)

# client = OpenAI(
#     api_key=''
# )

# openai.api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)

# Set your OpenAI API key
# openai.api_key = os.getenv('OPENAI_API_KEY')

def read_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.pdf':
            return read_pdf(file_path)
        elif file_extension == '.docx':
            return read_word(file_path)
        elif file_extension == '.csv':
            return read_csv(file_path)
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        raise RuntimeError(f"Failed to read file {file_path}: {str(e)}")

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        content = ""
        for page in reader.pages:
            content += page.extract_text()
        return content

def read_word(file_path):
    doc = docx.Document(file_path)
    content = "\n".join([para.text for para in doc.paragraphs])
    return content

def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_string()

def read_file_from_s3(bucket_name, file_key):
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket_name, file_key, '/tmp/' + file_key)
        file_path = '/tmp/' + file_key
        return read_file(file_path)
    except NoCredentialsError:
        raise RuntimeError("Credentials not available")
    except Exception as e:
        raise RuntimeError(f"Failed to read file from S3: {str(e)}")

def analyze_content_with_genai(prompt):    
    # print(prompt)
    response = client_genai.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt,
    )

    # Use the response as a JSON string.
    #print(response.text)
    return response.text

# def analyze_content_with_openai(content):
#     completion = client.chat.completions.create(
#         model="gemini-1.5-turbo",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are an agile product owner.",
#             },
#             {
#                 "role": "user",
#                 "content": f"Identify all data validation requirements in the following text:\n\n{content}",
#             },
#         ],
#         # max_tokens=500,
#         # temperature=0.5,
#     )
#     # response = openai.ChatCompletion.create(
#     #     model="gpt-4",
#     #     messages=[
#     #         {
#     #             "role": "system",
#     #             "content": "You are an agile product owner.",
#     #         },
#     #         {
#     #             "role": "user",
#     #             "content": f"Identify all data validation requirements in the following text:\n\n{content}",
#     #         },
#     #     ],
#     #     max_tokens=500,
#     #     temperature=0.5,
#     # )
#     # return response.choices[0].message['content']
#     return completion.choices[0].message.content

# def analyze_content_with_openai(content):
    # completion = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[
    #         {
    #             "role": "agile product owner",
    #             "content": f"Identify all data validation requirements in the following text:\n\n{content}",
    #         },
    #     ]
    # )
    # response = openai.Completion.create(
    #     engine="text-davinci-003",
    #     prompt=f"Identify all data validation requirements in the following text:\n\n{content}",
    #     max_tokens=500,
    #     n=1,
    #     stop=None,
    #     temperature=0.5,
    # )
    # return response.choices[0].text.strip()
    # return completion.choices[0].message['content']

@app.route('/read-file', methods=['POST'])
def read_file_endpoint():
    file_path = request.json.get('file_path')
    try:
        content = read_file(file_path)
        validation_requirements = analyze_content_with_genai
        (content)
        return jsonify({"content": content, "validation_requirements": validation_requirements})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/extract-regulatory-reporting-instructions', methods=['POST'])
def extractRegulatoryReportingInstructions():
    text = request.json.get('text_to_analyze')
    try:

        prompt = f"""
        Identify all data validation requirements in the following text:
        {text}
        
        Provide output in JSON format with fields like requirement and description
        """

        # Call the analyze_content_with_genai function to get the raw response
        reporting_instructions_raw = analyze_content_with_genai(prompt)
        
        # Parse the JSON-like string into a proper JSON object
        reporting_instructions = json.loads(reporting_instructions_raw.strip("```json\n").strip("```"))
        
        return jsonify({"content": text, "reporting_instructions": reporting_instructions})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/extract-data-validation-rules', methods=['POST'])
def extractDataValidationRulesFromRegulatoryReportingInstructions():
    requirements = request.json.get('reporting_instructions')  # List of data validation requirements
    try:
        # Prepare the prompt for Gemini AI
        prompt = f"""
        Based on the following data validation requirements:
        {json.dumps(requirements, indent=2)}
        
        Provide detailed data validation rules for each requirement in JSON format. Each rule should include:
        - `requirement`: The original requirement.
        - `field`: The field to validate.
        - `condition`: The condition to check (e.g., "field must be greater than 0").
        - `description`: A description of the rule.
        """
        print(prompt)

        data_validation_rules_raw = analyze_content_with_genai(prompt)
        
        # Parse the response text into JSON
        rules_raw = data_validation_rules_raw.strip("```json\n").strip("```")
        validation_rules = json.loads(rules_raw)
        
        return jsonify({"data_validation_rules": validation_rules})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/generate-datavalidation-functions', methods=['POST'])
def generateDataValidationFunctions():
    data_validation_rules = request.json.get('data_validation_rules')
    try:
        prompt = f"""
        Based on the following data validation rules:
        {json.dumps(data_validation_rules, indent=2)}
        
        Generate Python functions for each rule. Each function should:
        - Take a single record (dictionary) as input.
        - Return True if the record satisfies the rule, otherwise False.
        Provide the output in JSON format with fields:
        - `requirement`: The original requirement.
        - `function`: The Python function as a string.
        """

        data_validation_functions_raw = analyze_content_with_genai(prompt)

        functions_raw = data_validation_functions_raw.strip("```json\n").strip("```")
        generated_functions = json.loads(functions_raw)
        
        return jsonify({"data_validation_functions": generated_functions})
    except Exception as e:
        return jsonify({"error": str(e)}), 400



@app.route('/validate-dataset-with-generated-functions', methods=['POST'])
def generateAndApplyValidationFunctions():
    try:
        # Get the dataset and validation rules from the request
        dataset = request.json.get('dataset')  # List of records (e.g., rows in a table)
        generated_functions = request.json.get('data_validation_functions')

        # Apply the generated functions to the dataset
        validation_results = []
        for func in generated_functions:
            requirement = func['requirement']
            function_code = func['function']

            # Dynamically compile the function
            exec(function_code, globals())

            # Extract the function name from the code
            function_name = function_code.split('def ')[1].split('(')[0]

            # Apply the function to each record in the dataset
            for record in dataset:
                result = eval(f"{function_name}(record)")
                validation_results.append({
                    "record": record,
                    "requirement": requirement,
                    "result": result
                })

        return jsonify({"validation_results": validation_results})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/read-file-s3', methods=['POST'])
def read_file_s3_endpoint():
    bucket_name = request.json.get('bucket_name')
    file_key = request.json.get('file_key')
    try:
        content = read_file_from_s3(bucket_name, file_key)
        validation_requirements = analyze_content_with_genai(content)
        return jsonify({"content": content, "validation_requirements": validation_requirements})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)