from __future__ import annotations
import base64
import os
import json
import sys
import requests
from typing import List, Dict, Any, Optional
from datetime import date, datetime
from langchain.chains import TransformChain
from langchain_core.messages import HumanMessage
from langchain_core.runnables import chain
# Update imports to use Pydantic directly as per the warning
from pydantic import BaseModel, Field
import glob
import pandas as pd
import shutil
import argparse
from prettytable import PrettyTable
from tabulate import tabulate
import colorama
from colorama import Fore, Style
from openai import OpenAI
import traceback

# Import API key from keys.py or environment variable
try:
    from keys import OPENROUTER_API_KEY
except ImportError:
    print(f"{Fore.YELLOW}Warning: keys.py not found, trying to use environment variable{Style.RESET_ALL}")
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        print(f"{Fore.RED}Error: OPENROUTER_API_KEY not found in keys.py or environment variables{Style.RESET_ALL}")
        sys.exit(1)

# Initialize colorama for colored terminal output
colorama.init()

# Test API key and connectivity
def test_api_connectivity():
    print(f"{Fore.CYAN}Testing OpenRouter API connectivity...{Style.RESET_ALL}")
    
    # Check if API key is set
    if not OPENROUTER_API_KEY:
        print(f"{Fore.RED}Error: OpenRouter API key is not set{Style.RESET_ALL}")
        return False
    
    # Hide most of the API key for display
    masked_key = OPENROUTER_API_KEY[:4] + "..." + OPENROUTER_API_KEY[-4:] if len(OPENROUTER_API_KEY) > 8 else "****"
    print(f"{Fore.YELLOW}Using API key: {masked_key}{Style.RESET_ALL}")
    
    # Test API connection with a simple request
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )
        
        # Test with a simple models list request
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
        )
        
        if response.status_code == 200:
            print(f"{Fore.GREEN}API connection successful!{Style.RESET_ALL}")
            
            # Check if our specific model is available
            models = response.json().get("data", [])
            model_names = [model.get("id") for model in models]
            
            if "google/gemini-flash-1.5-8b-exp" in model_names:
                print(f"{Fore.GREEN}Model 'google/gemini-flash-1.5-8b-exp' is available{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Model 'google/gemini-flash-1.5-8b-exp' not found in available models{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Available models: {model_names}{Style.RESET_ALL}")
            return True
        else:
            print(f"{Fore.RED}API connection failed with status code: {response.status_code}{Style.RESET_ALL}")
            print(f"{Fore.RED}Response: {response.text}{Style.RESET_ALL}")
            return False
    except Exception as e:
        print(f"{Fore.RED}Error testing API connectivity: {str(e)}{Style.RESET_ALL}")
        traceback.print_exc()
        return False

# Updated class for OpenRouter API using OpenAI client
class GeminiFlashAPI:
    def __init__(self, api_key: str, debug_mode: bool = False):
        self.api_key = api_key
        self.model = "google/gemini-flash-1.5-8b-exp"
        self.debug_mode = debug_mode
        
        # Try alternative models if specified model doesn't work
        self.backup_models = [
            "anthropic/claude-3-haiku-20240307",
            "google/gemini-pro-1.5-flash",
            "mistralai/mistral-7b-instruct"
        ]
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.extra_headers = {
            "HTTP-Referer": "http://localhost:3000",  # Replace with your site
            "X-Title": "Medical Prescription Parser"
        }
    
    def debug_print(self, message):
        if self.debug_mode:
            print(f"{Fore.BLUE}[DEBUG] {message}{Style.RESET_ALL}")
    
    def generate(self, messages: List[Dict[str, Any]], temperature: float = 0.7, 
                 max_retries: int = 3, try_backup_models: bool = True) -> str:
        """Generate a response from Gemini Flash via OpenRouter API with retries and fallbacks"""
        
        models_to_try = [self.model] + (self.backup_models if try_backup_models else [])
        
        for i, current_model in enumerate(models_to_try):
            for attempt in range(max_retries):
                try:
                    self.debug_print(f"Attempt {attempt+1}/{max_retries} with model: {current_model}")
                    
                    # For direct API call with more debugging
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "http://localhost:3000",
                        "X-Title": "Medical Prescription Parser"
                    }
                    
                    payload = {
                        "model": current_model,
                        "messages": messages,
                        "temperature": temperature
                    }
                    
                    # Use requests for more control and better error messages
                    self.debug_print(f"Sending request payload: {json.dumps(payload)[:500]}...")
                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        self.debug_print(f"Received response: {json.dumps(result)[:500]}...")
                        
                        if result and "choices" in result and len(result["choices"]) > 0:
                            content = result["choices"][0]["message"]["content"]
                            if content:
                                print(f"{Fore.GREEN}Successfully received content from {current_model}{Style.RESET_ALL}")
                                return content
                    
                    print(f"{Fore.RED}API call failed with status code: {response.status_code}{Style.RESET_ALL}")
                    print(f"{Fore.RED}Response: {response.text}{Style.RESET_ALL}")
                    
                except Exception as e:
                    print(f"{Fore.RED}API Request Error (attempt {attempt+1}): {str(e)}{Style.RESET_ALL}")
                    traceback.print_exc()
                
                print(f"{Fore.YELLOW}Retrying in 2 seconds...{Style.RESET_ALL}")
                import time
                time.sleep(2)  # Wait between retries
            
            if i < len(models_to_try) - 1:
                print(f"{Fore.YELLOW}Switching to backup model: {models_to_try[i+1]}{Style.RESET_ALL}")
        
        print(f"{Fore.RED}All models and retries failed{Style.RESET_ALL}")
        return None

class MedicationItem(BaseModel):
    name: str
    dosage: str
    frequency: str
    duration: str

class PrescriptionInformations(BaseModel):
    """Information about an image."""
    patient_name: str = Field(description="Patient's name")
    patient_age: int = Field(description="Patient's age")
    patient_gender: str = Field(description="Patient's gender")
    doctor_name: str = Field(description="Doctor's name")
    doctor_license: str = Field(description="Doctor's license number")
    prescription_date: datetime = Field(description="Date of the prescription")
    medications: List[MedicationItem] = []
    additional_notes: str = Field(description="Additional notes or instructions")

def load_images(inputs: dict) -> dict:
    """Load images from files and encode them as base64."""
    image_paths = inputs["image_paths"]

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    images_base64 = [encode_image(image_path) for image_path in image_paths]
    return {"images": images_base64}

def validate_image(image_path: str) -> bool:
    """Validate that the image exists and is a supported format."""
    if not os.path.exists(image_path):
        print(f"{Fore.RED}Error: Image file not found at {image_path}{Style.RESET_ALL}")
        return False
    
    # Check file extension
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    file_ext = os.path.splitext(image_path)[1].lower()
    
    if file_ext not in valid_extensions:
        print(f"{Fore.RED}Error: Unsupported image format. Supported formats: {valid_extensions}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Consider converting your image to JPG or PNG format.{Style.RESET_ALL}")
        return False
    
    # Check file size
    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    if file_size_mb > 20:  # 20MB limit as an arbitrary limit
        print(f"{Fore.RED}Warning: Image file is very large ({file_size_mb:.2f}MB).{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Consider compressing the image for better performance.{Style.RESET_ALL}")
    
    return True

def process_prescription(image_paths: List[str], debug_mode: bool = False) -> dict:
    """Process prescription images and extract information."""
    # Load images
    try:
        images_data = load_images({"image_paths": image_paths})
        print(f"{Fore.GREEN}Successfully loaded {len(images_data['images'])} images{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error loading images: {str(e)}{Style.RESET_ALL}")
        traceback.print_exc()
        return create_default_response(f"Error loading images: {str(e)}")
    
    # Initialize Gemini Flash API client
    gemini_api = GeminiFlashAPI(api_key=OPENROUTER_API_KEY, debug_mode=debug_mode)
    
    # Prepare messages for OpenRouter
    system_message = {
        "role": "system",
        "content": """You are an expert medical transcriptionist specializing in deciphering and accurately transcribing handwritten medical prescriptions. Extract all relevant information with the highest degree of precision and return it in a structured JSON format."""
    }
    
    # Create content for the user message
    user_content = [
        {
            "type": "text",
            "text": """
            Analyze this prescription image and extract the following information:
            - Patient's full name
            - Patient's age (in years)
            - Patient's gender
            - Doctor's full name
            - Doctor's license number
            - Prescription date (in YYYY-MM-DD format)
            - List of medications including:
              - Medication name
              - Dosage
              - Frequency
              - Duration
            - Additional notes or instructions
            
            Format your response as a valid JSON object with the following structure:
            {
                "patient_name": "string",
                "patient_age": integer,
                "patient_gender": "string",
                "doctor_name": "string",
                "doctor_license": "string",
                "prescription_date": "YYYY-MM-DD",
                "medications": [
                    {
                        "name": "string",
                        "dosage": "string",
                        "frequency": "string",
                        "duration": "string"
                    }
                ],
                "additional_notes": "string"
            }
            
            If any information is not legible or missing, use empty strings or appropriate default values.
            The response should be ONLY the JSON object with no additional text.
            """
        }
    ]
    
    # Add images to content
    for img in images_data["images"]:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img}"}
        })
    
    # Create messages
    messages = [
        system_message,
        {"role": "user", "content": user_content}
    ]
    
    # Generate response
    try:
        print(f"{Fore.YELLOW}Sending request to OpenRouter API...{Style.RESET_ALL}")
        response = gemini_api.generate(
            messages=messages, 
            temperature=0.5, 
            max_retries=3, 
            try_backup_models=True
        )
        
        if not response:
            print(f"{Fore.RED}Error: No response received from API{Style.RESET_ALL}")
            return create_default_response("Error: Failed to process prescription - API returned empty response")
        
        print(f"{Fore.GREEN}Response received. Processing JSON...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Raw response:{Style.RESET_ALL}\n{response[:500]}...")
        
        # Extract JSON from the response
        json_str = response
        # Sometimes the model might wrap the JSON in ```json ... ``` markers
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
            
        # Parse the JSON
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"{Fore.RED}JSON parsing error: {str(e)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Attempting to fix malformed JSON...{Style.RESET_ALL}")
            
            # Basic attempt to fix common JSON issues
            json_str = json_str.replace("'", "\"")
            json_str = json_str.replace("\n", " ")
            
            try:
                result = json.loads(json_str)
            except:
                print(f"{Fore.RED}Failed to fix JSON. Returning default values.{Style.RESET_ALL}")
                return create_default_response(f"Error parsing JSON response: {json_str[:100]}...")
        
        # Convert date string to datetime object
        if "prescription_date" in result and result["prescription_date"]:
            try:
                result["prescription_date"] = datetime.strptime(result["prescription_date"], "%Y-%m-%d")
            except:
                print(f"{Fore.YELLOW}Warning: Couldn't convert date format, keeping as string{Style.RESET_ALL}")
                # Keep as string if conversion fails
        
        # Validate the structure of the result
        expected_keys = ["patient_name", "patient_age", "patient_gender", "doctor_name", 
                         "doctor_license", "prescription_date", "medications", "additional_notes"]
        for key in expected_keys:
            if key not in result:
                print(f"{Fore.YELLOW}Warning: Missing expected key '{key}' in result, adding default{Style.RESET_ALL}")
                if key == "medications":
                    result[key] = []
                elif key == "patient_age":
                    result[key] = 0
                else:
                    result[key] = ""
        
        return result
    except Exception as e:
        print(f"{Fore.RED}Error processing response: {str(e)}{Style.RESET_ALL}")
        print(f"{Fore.RED}Traceback:{Style.RESET_ALL}")
        traceback.print_exc()
        return create_default_response(f"Error: {str(e)}")

def create_default_response(error_message: str = "Unknown error") -> dict:
    """Create a default response structure with error message."""
    return {
        "patient_name": "",
        "patient_age": 0,
        "patient_gender": "",
        "doctor_name": "",
        "doctor_license": "",
        "prescription_date": datetime.now(),
        "medications": [],
        "additional_notes": error_message
    }

def remove_temp_folder(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains

def print_header(text):
    """Print a formatted header"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}")
    print("=" * 50)

def print_section(section_name):
    """Print a section name"""
    print(f"\n{Fore.GREEN}{Style.BRIGHT}{section_name}{Style.RESET_ALL}")
    print("-" * 40)

def format_additional_notes(notes):
    """Format additional notes as bullet points"""
    if not notes:
        return ""
    if isinstance(notes, list):
        return "\n".join([f"• {note}" for note in notes])
    else:
        return "\n".join([f"• {line}" for line in notes.split("\n") if line.strip()])

def save_to_file(data, filename="prescription_results.json"):
    """Save prescription data to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n{Fore.YELLOW}Results saved to {filename}{Style.RESET_ALL}")

def main():
    # Set up argument parser
    arg_parser = argparse.ArgumentParser(description='Medical Prescription Image Parser')
    arg_parser.add_argument('--image', '-i', required=True, help='Path to the prescription image file')
    arg_parser.add_argument('--output', '-o', default="prescription_results.json", help='Output JSON file path')
    arg_parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode with more verbose output')
    arg_parser.add_argument('--test-api', '-t', action='store_true', help='Test API connectivity and exit')
    args = arg_parser.parse_args()
    
    print_header("MEDICAL PRESCRIPTION PARSER")
    
    # Test API connectivity if requested
    if args.test_api:
        test_api_connectivity()
        sys.exit(0)
    
    image_path = args.image
    output_file = args.output
    debug_mode = args.debug
    
    if debug_mode:
        print(f"{Fore.YELLOW}Debug mode is enabled{Style.RESET_ALL}")
    
    # Validate image
    if not validate_image(image_path):
        return
    
    print(f"Processing image: {Fore.YELLOW}{image_path}{Style.RESET_ALL}")
    
    # Before proceeding with processing, test API connectivity
    if not test_api_connectivity():
        print(f"{Fore.RED}API connectivity test failed, but attempting to process anyway...{Style.RESET_ALL}")
    
    # Create temp output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(image_path).split('.')[0].replace(' ', '_')
    output_folder = os.path.join(".", f"Check_{filename}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    
    # Copy image to temp folder
    temp_image_path = os.path.join(output_folder, os.path.basename(image_path))
    shutil.copy2(image_path, temp_image_path)
    
    print(f"\n{Fore.YELLOW}Processing prescription...{Style.RESET_ALL}")
    
    try:
        # Process the prescription image
        final_result = process_prescription([temp_image_path], debug_mode=debug_mode)
        
        # Display results
        print_header("PRESCRIPTION INFORMATION")
        
        # Basic patient and doctor info
        table = PrettyTable()
        table.field_names = ["Field", "Value"]
        table.align["Field"] = "l"
        table.align["Value"] = "l"
        
        for key, value in final_result.items():
            if key != 'medications' and key != 'additional_notes':
                table.add_row([key.replace('_', ' ').title(), value])
        
        print(table)
        
        # Medications
        if 'medications' in final_result and final_result['medications']:
            print_section("MEDICATIONS")
            med_table = PrettyTable()
            med_table.field_names = ["Name", "Dosage", "Frequency", "Duration"]
            med_table.align = "l"
            
            for med in final_result['medications']:
                med_table.add_row([med.get('name', ''), med.get('dosage', ''), 
                med.get('frequency', ''), med.get('duration', '')])
            
            print(med_table)
        
        # Additional notes
        if 'additional_notes' in final_result and final_result['additional_notes']:
            print_section("ADDITIONAL NOTES")
            formatted_notes = format_additional_notes(final_result['additional_notes'])
            print(formatted_notes)
        
        # Save to file
        save_to_file(final_result, output_file)
        
    except Exception as e:
        print(f"{Fore.RED}Error processing prescription: {str(e)}{Style.RESET_ALL}")
        traceback.print_exc()
    finally:
        # Clean up temp folder
        remove_temp_folder(output_folder)
        print(f"\n{Fore.GREEN}Process completed!{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
    
    
# run the code with debugging mode:
# python prescription.py --image C:\Users\Hp\Downloads\pres.jpg --output results.json --debug

# To test only API connectivity:
# python prescription.py --test-api

# For normal usage:
# python prescription.py --image C:\Users\Hp\Downloads\pres.jpg --output results.json