from src.job_matcher import JobMatcher
from src.parser import extract_text

resume = extract_text("data/resumes/Shivani_Pattnayak_RESUME (2).pdf")

matcher = JobMatcher()

jobs = matcher.find_matching_jobs(
    resume,
    top_k=5
)

for i, job in enumerate(jobs, start=1):

    print("=" * 60)

    print(f"Rank #{i}")

    print("Match Score :", job["match_score"])

    print("Title       :", job["title"])

    print("Company     :", job["companyName"])

    print("Experience  :", job["experience"])

    print("Location    :", job["location"])