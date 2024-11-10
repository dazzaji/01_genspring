

# Create and activate virtual environment
# python -m venv venv
# source venv/bin/activate  # On Windows use: venv\Scripts\activate

# pip install flask google-generativeai python-dotenv pydantic
# or
# python -m pip install -r requirements.txt

# RUN: python app.py


import os
from dotenv import load_dotenv
import json
import sys
import subprocess
from typing import Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import google.generativeai as genai
from google.api_core import retry
from google.api_core import exceptions as google_exceptions
import contextlib
from flask import Flask, render_template, request, jsonify, session
from functools import wraps

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = os.urandom(24)

@contextlib.contextmanager
def tee_output(filename=None):
    """
    Context manager that captures all stdout/stderr and writes it to a file while
    still displaying it in the terminal.
    """
    if filename is None:
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{script_name}_{timestamp}_output.log"
    
    try:
        process = subprocess.Popen(
            ['tee', filename],
            stdin=subprocess.PIPE,
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = process.stdin
        
        yield process  # Return process so we can access the log file name
        
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        if process.stdin:
            process.stdin.close()
        process.wait()

# Load environment variables
load_dotenv()

# Access API key from environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Initialize Gemini client
genai.configure(api_key=gemini_api_key)

# Define Pydantic model for initial plan structure
class InitialPlanStructure(BaseModel):
    """
    Model for validating the initial plan structure (outline only).
    """
    Title: str
    Overall_Summary: str
    Original_Goal: str
    Detailed_Outline: list[Dict[str, str]] = Field(..., description="List of steps with content")
    Evaluation_Criteria: Dict[str, str] = Field(..., description="Criteria for evaluating each step")
    Success_Measures: list[str]

def generate_initial_structure(goal: str) -> Optional[Dict]:
    """
    Generates the initial plan structure using Gemini Pro.
    Returns a validated plan structure dictionary or None if generation fails.
    """
    prompt = f"""
    You are a top consultant developing an initial project outline.
    Create a high-level project structure in JSON format for the following goal: {goal}

    Focus ONLY on creating a concise outline structure with:
    - A clear title
    - Brief overall summary
    - Key steps (5-7 maximum)
    - Basic evaluation criteria
    - Success measures

    The JSON must strictly follow this template:
    {{
      "Title": "...",
      "Overall_Summary": "...",
      "Original_Goal": "{goal}",
      "Detailed_Outline": [
        {{"name": "Step 1", "content": "Brief description of step 1"}},
        {{"name": "Step 2", "content": "Brief description of step 2"}},
        ...
      ],
      "Evaluation_Criteria": {{
        "Step 1": "Criteria for evaluating Step 1",
        "Step 2": "Criteria for evaluating Step 2",
        ...
      }},
      "Success_Measures": [
        "Success measure 1",
        "Success measure 2",
        ...
      ]
    }}

    Keep all descriptions and content brief and focused on structure only.
    Your response must be valid JSON only, with no text outside the JSON structure.
    """

    model = genai.GenerativeModel('gemini-1.5-pro',
                                generation_config={
                                    "temperature": 0.1,
                                    "top_p": 1,
                                    "top_k": 1,
                                    "max_output_tokens": 4096
                                })

    @retry.Retry(predicate=retry.if_exception_type(
        google_exceptions.ResourceExhausted,
        google_exceptions.ServiceUnavailable,
        google_exceptions.DeadlineExceeded,
        google_exceptions.InternalServerError
    ), deadline=90)
    def generate_with_retry():
        return model.generate_content(prompt)

    try:
        print("\nGenerating initial plan structure...")
        response = generate_with_retry()
        
        try:
            # Extract and parse JSON
            json_start = response.text.find('{')
            json_end = response.text.rfind('}') + 1
            json_str = response.text[json_start:json_end]
            plan_dict = json.loads(json_str)
            
            # Ensure Evaluation_Criteria is a dictionary
            if isinstance(plan_dict["Evaluation_Criteria"], list):
                plan_dict["Evaluation_Criteria"] = {
                    item["name"]: item["criteria"] 
                    for item in plan_dict["Evaluation_Criteria"]
                }
            
            # Validate using Pydantic model
            validated_plan = InitialPlanStructure(**plan_dict)
            return validated_plan.model_dump()

        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON response from Gemini: {e}")
            logging.error(f"Raw response: {response.text}")
            return None
            
    except Exception as e:
        logging.error(f"Error generating plan structure: {type(e).__name__}: {e}")
        return None

def save_json_file(content: Dict, filename: str) -> bool:
    """
    Saves dictionary content to a JSON file.
    Returns True if successful, False otherwise.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2)
        print(f"\nSuccessfully saved: {filename}")
        return True
    except Exception as e:
        logging.error(f"Error saving {filename}: {e}")
        return False

@app.route('/')
def index():
    return render_template('input_goal.html')

@app.route('/generate_plan', methods=['POST'])
def generate_plan():
    with tee_output() as process:
        print("\nWelcome to the Initial Project Plan Structure Generator!")
        print("This module will create the basic structure for your project plan.")
        
        goal = request.form.get('goal')
        if not goal:
            print("Error: No goal provided")
            return jsonify({'error': 'No goal provided'}), 400
        
        print(f"\nReceived Project Goal: {goal}")
        
        try:
            plan = generate_initial_structure(goal)
            if plan:
                print("\nGenerated Plan Structure:")
                print(json.dumps(plan, indent=2))
                
                # Debug output for evaluation criteria
                print("\nEvaluation Criteria:")
                print(json.dumps(plan.get('Evaluation_Criteria', {}), indent=2))
                
                # Store the log filename in session for reference
                if hasattr(process, 'name'):
                    session['log_file'] = process.name
                
                session['current_plan'] = plan
                return render_template('edit_plan.html', 
                                     plan=plan,
                                     debug_eval=json.dumps(plan.get('Evaluation_Criteria', {}), indent=2))
            else:
                error_msg = "Failed to generate plan"
                print(f"\nError: {error_msg}")
                return jsonify({'error': error_msg}), 500
                
        except Exception as e:
            error_msg = str(e)
            print(f"\nError in generate_plan: {error_msg}")
            logging.error(f"Error in generate_plan: {e}")
            return jsonify({'error': error_msg}), 500

@app.route('/save_plan', methods=['POST'])
def save_plan():
    with tee_output():
        plan = request.json
        try:
            # Validate the edited plan
            validated_plan = InitialPlanStructure(**plan)
            plan_dict = validated_plan.model_dump()
            
            # Save both regular and timestamped versions
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            success = save_json_file(plan_dict, "plan_structure.json")
            if success:
                save_json_file(plan_dict, f"plan_structure_{timestamp}.json")
                print("\nYou can now run 1_IngestGoal-to-Plan.py to develop the full plan.")
                return jsonify({
                    'message': 'Plan saved successfully',
                    'next_step': 'You can now run 1_IngestGoal-to-Plan.py to develop the full plan.'
                })
            else:
                error_msg = "Failed to save plan structure"
                print(f"\nError: {error_msg}")
                return jsonify({'error': error_msg}), 500
                
        except Exception as e:
            error_msg = str(e)
            print(f"\nError saving plan: {error_msg}")
            logging.error(f"Error saving plan: {e}")
            return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Use port 5001 instead of default 5000