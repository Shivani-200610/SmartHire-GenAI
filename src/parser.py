import os
import json

from pypdf import PdfReader
from docx import Document
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


def extract_text(file_path: str) -> str:
    """Extract text from PDF or DOCX."""

    ext = os.path.splitext(file_path)[1].lower()

    text = ""

    if ext == ".pdf":
        reader = PdfReader(file_path)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    elif ext == ".docx":
        doc = Document(file_path)

        for para in doc.paragraphs:
            text += para.text + "\n"

    else:
        raise ValueError("Unsupported file format")

    return text


def parse_resume(text: str):

    prompt = f"""
You are an expert ATS Resume Parser.

Extract the following information.

Return ONLY valid JSON.

{{
"name":"",
"email":"",
"phone":"",
"education":[],
"experience":[],
"skills":[],
"projects":[],
"certifications":[],
"target_role":""
}}

Resume

{text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    result = response.text.strip()

    result = result.replace("```json", "")
    result = result.replace("```", "").strip()

    return json.loads(result)