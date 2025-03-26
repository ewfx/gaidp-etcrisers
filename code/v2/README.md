# My Python App

This project is a simple Flask application that provides REST endpoints to read file content and raw text input.

## Project Structure

```
my-python-app
├── src
│   ├── app.py                # Entry point of the application
│   ├── controllers           # Contains business logic
│   │   └── __init__.py
│   ├── routes                # Defines REST endpoints
│   │   └── __init__.py
│   └── services              # Additional service logic (currently empty)
│       └── __init__.py
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Requirements

To run this application, you need to have Python and Flask installed. You can install the required packages using:

```
pip install -r requirements.txt
```

## Running the Application

To start the Flask application, navigate to the `src` directory and run:

```
python app.py
```

The application will be available at `http://127.0.0.1:5000/`.

## Endpoints

1. **Read File Content**
   - **Endpoint:** `/read-file`
   - **Method:** POST
   - **Request Body:** JSON with a key `file_path` containing the path to the file.
   - **Response:** The content of the file.

2. **Read Raw Text**
   - **Endpoint:** `/read-text`
   - **Method:** POST
   - **Request Body:** JSON with a key `text` containing the raw text.
   - **Response:** The raw text input. 

## License

This project is licensed under the MIT License.