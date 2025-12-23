# Handwritten Medical Prescription Reader

**Overview**  
This module automatically extracts and structures data from handwritten medical prescriptions. It validates fields such as patient details, medications, and dosages, and outputs structured JSON while handling common transcription errors.

**Technologies Used**
- Python
- OpenRouter API (Gemini Flash 1.5)
- Pydantic

**Features**
- Multi-model fallback for robust data extraction
- Error handling for common transcription mistakes
- Reduces manual data entry time by ~60% for medical staff

**Usage**
1. Provide scanned images of prescriptions.
2. Run the script to get structured JSON output.
