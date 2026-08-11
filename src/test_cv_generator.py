from src.cv_generator import generate_cv_suggestions


resume_text = """
Shivani is a Computer Science Engineering student specializing
in Artificial Intelligence.

Skills:
Python, Java, Machine Learning, Deep Learning, Streamlit

Projects:
Chest X-ray disease detection using deep learning.
Social network friend suggestion system.

Experience:
Generative AI internship.
"""


job = {
    "title": "AI Engineering / Research Intern",
    "companyName": "Brainmage AI",
    "tagsAndSkills": "Python, Machine Learning, Deep Learning, RAG, LangChain",
    "jobDescription": """
    Work on AI and machine learning applications.
    Experience with Python, deep learning, RAG and LLM applications
    is preferred.
    """
}


result = generate_cv_suggestions(resume_text, job)

print("\n==============================")
print("CV IMPROVEMENT SUGGESTIONS")
print("==============================")

print("\nMissing Skills:")
for skill in result.get("missing_skills", []):
    print("-", skill)

print("\nWeak Bullets:")

for bullet in result.get("weak_bullets", []):
    print("\nOriginal:")
    print(bullet.get("original", ""))

    print("\nImproved:")
    print(bullet.get("improved", ""))

print("\nRewritten Summary:")
print(result.get("rewritten_summary", ""))

print("\nOverall Suggestions:")
for suggestion in result.get("overall_suggestions", []):
    print("-", suggestion)