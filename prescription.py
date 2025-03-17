from __future__ import annotations
import base64
import os
import json
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
from keys import OPENROUTER_API_KEY

# Initialize colorama for colored terminal output
colorama.init()

# Set up the OpenRouter API key
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

# Updated class for OpenRouter API using OpenAI client
class GeminiFlashAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "google/gemini-flash-1.5-8b-exp"
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.extra_headers = {
            "HTTP-Referer": "http://localhost:3000",  # Replace with your site
            "X-Title": "Medical Prescription Parser"
        }
    
    def generate(self, messages: List[Dict[str, Any]], temperature: float = 0.7) -> str:
        """Generate a response from Gemini Flash via OpenRouter API"""
        completion = self.client.chat.completions.create(
            extra_headers=self.extra_headers,
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        
        return completion.choices[0].message.content

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

load_images_chain = TransformChain(
    input_variables=["image_paths"],
    output_variables=["images"],
    transform=load_images
)

def process_prescription(image_paths: List[str]) -> dict:
    """Process prescription images and extract information."""
    # Load images
    images_data = load_images({"image_paths": image_paths})
    
    # Initialize Gemini Flash API client
    gemini_api = GeminiFlashAPI(api_key=OPENROUTER_API_KEY)
    
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
            "image_url": {"url": f"data:image/png;base64,{img}"}
        })
    
    # Create messages
    messages = [
        system_message,
        {"role": "user", "content": user_content}
    ]
    
    # Generate response
    try:
        response = gemini_api.generate(messages, temperature=0.5)
        
        # Extract JSON from the response
        json_str = response
        # Sometimes the model might wrap the JSON in ```json ... ``` markers
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
            
        # Parse the JSON
        result = json.loads(json_str)
        
        # Convert date string to datetime object
        if "prescription_date" in result and result["prescription_date"]:
            try:
                result["prescription_date"] = datetime.strptime(result["prescription_date"], "%Y-%m-%d")
            except:
                pass  # Keep as string if conversion fails
        
        return result
    except Exception as e:
        print(f"{Fore.RED}Error processing response: {str(e)}{Style.RESET_ALL}")
        raise

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
    args = arg_parser.parse_args()
    
    image_path = args.image
    output_file = args.output
    
    if not os.path.exists(image_path):
        print(f"{Fore.RED}Error: Image file not found at {image_path}{Style.RESET_ALL}")
        return
    
    print_header("MEDICAL PRESCRIPTION PARSER")
    print(f"Processing image: {Fore.YELLOW}{image_path}{Style.RESET_ALL}")
    
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
        final_result = process_prescription([temp_image_path])
        
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
    
    # Clean up temp folder
    remove_temp_folder(output_folder)
    print(f"\n{Fore.GREEN}Process completed!{Style.RESET_ALL}")

if __name__ == "__main__":
    main()