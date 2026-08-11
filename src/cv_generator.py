import os
import json

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


def generate_cv_suggestions(resume_text: str, job: dict):
    """
    Generate CV improvement suggestions based on
    the candidate's resume and a selected job.
    """

    job_title = job.get("title", "")
    company = job.get("companyName", "")
    skills = job.get("tagsAndSkills", "")
    description = job.get("jobDescription", "")

    prompt = f"""
You are an expert AI career coach and professional resume reviewer.

Analyze the candidate's resume against the target job.

Return ONLY valid JSON.

Use this exact structure:

{{
    "missing_skills": [],
    "weak_bullets": [
        {{
            "original": "",
            "improved": ""
        }}
    ],
    "rewritten_summary": "",
    "overall_suggestions": []
}}

IMPORTANT:
- Do not invent experience that is not present in the resume.
- Missing skills should be skills relevant to the job that are absent
  or not clearly demonstrated in the resume.
- Weak bullets should only be identified when there is an actual
  experience/project bullet that could be improved.
- Improved bullets must remain truthful to the original experience.
- The rewritten summary must be based only on the candidate's
  actual background.
- Keep suggestions specific and actionable.

TARGET JOB
-----------
Title: {job_title}
Company: {company}

Required Skills:
{skills}

Job Description:
{description}

CANDIDATE RESUME
----------------
{resume_text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    result = response.text.strip()

    # Remove Markdown code fences if Gemini adds them
    result = result.replace("```json", "")
    result = result.replace("```", "").strip()

    return json.loads(result)