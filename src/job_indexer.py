import pandas as pd

df = pd.read_excel("data/jobs/jobs_data.xlsx")

print("Dataset Shape:", df.shape)

# Keep useful columns
df = df[
    [
        "title",
        "companyName",
        "tagsAndSkills",
        "experience",
        "location",
        "jobDescription",
    ]
]

df = df.fillna("")

print(df.head())

# Build searchable text
df["search_text"] = (
    "Title: " + df["title"] +
    "\nCompany: " + df["companyName"] +
    "\nSkills: " + df["tagsAndSkills"] +
    "\nExperience: " + df["experience"] +
    "\nLocation: " + df["location"] +
    "\nDescription: " + df["jobDescription"]
)

print(df["search_text"][0])