from pydantic import BaseModel
from typing import List


class ResumeProfile(BaseModel):
    name: str
    email: str
    phone: str
    education: List[str]
    experience: List[str]
    skills: List[str]
    projects: List[str]
    certifications: List[str]
    target_role: str