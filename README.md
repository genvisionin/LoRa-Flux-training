# Project Setup Guide
## Prerequisites
Ensure you have the following installed on your system:

Python (latest stable version)
Git
## Installation Steps
### 1. Clone the Repository
First, clone the ai-toolkit repository:

bash
Copy
Edit
git clone <repository_url>
Navigate into the cloned repository:

bash
Copy
Edit
cd ai-toolkit
### 2. Set Up a Virtual Environment
Create and activate a virtual environment:

On Windows:
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate
On macOS/Linux:
bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
### 3. Install Dependencies
Run the following command to install all required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
### 4. Run the Application
Ensure that app.py is present inside the ai-toolkit directory. Then, start the application by running:

bash
Copy
Edit
python app.py

## Testing the APIs
To test the application, execute the test scripts:

bash
Copy
Edit
python testApis.py
