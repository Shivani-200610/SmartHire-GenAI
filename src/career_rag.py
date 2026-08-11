import os
import glob

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai

from src.guardrails import (
    validate_question,
    validate_answer
)


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()


# ==========================================
# CONFIGURATION
# ==========================================

CAREER_NOTES_DIR = "data/career_notes"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# GEMINI CLIENT
# ==========================================

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ==========================================
# LOAD CAREER KNOWLEDGE
# ==========================================

def load_career_notes():
    """
    Load all career knowledge files from
    data/career_notes.
    """

    documents = []

    files = glob.glob(
        os.path.join(
            CAREER_NOTES_DIR,
            "*.txt"
        )
    )

    for file_path in files:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read()

        documents.append(
            {
                "source": os.path.basename(
                    file_path
                ),
                "text": text
            }
        )

    return documents


# ==========================================
# CREATE TEXT CHUNKS
# ==========================================

def create_chunks(documents):
    """
    Create overlapping chunks from career
    knowledge documents.

    Overlapping chunks help keep headings
    together with their related content.
    """

    chunks = []

    # Number of words in each chunk
    chunk_size = 180

    # Number of words shared between
    # consecutive chunks
    overlap = 40

    for document in documents:

        text = document["text"]
        source = document["source"]

        # Normalize line endings
        text = text.replace(
            "\r\n",
            "\n"
        )

        text = text.strip()

        words = text.split()

        if not words:
            continue

        start = 0

        while start < len(words):

            end = min(
                start + chunk_size,
                len(words)
            )

            chunk = " ".join(
                words[start:end]
            )

            if chunk.strip():

                chunks.append(
                    {
                        "text": chunk,
                        "source": source
                    }
                )

            # Stop when the entire document
            # has been processed
            if end >= len(words):
                break

            # Move forward while keeping
            # some content from the previous
            # chunk
            start = end - overlap

    return chunks


# ==========================================
# BUILD CAREER VECTOR INDEX
# ==========================================

def build_career_index():

    documents = load_career_notes()

    if not documents:

        raise ValueError(
            "No career notes found in "
            "data/career_notes"
        )

    chunks = create_chunks(
        documents
    )

    if not chunks:

        raise ValueError(
            "No career knowledge chunks "
            "were created."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Generate embeddings
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    )

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    # Inner Product + normalized embeddings
    # = cosine similarity
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index, chunks


# ==========================================
# RETRIEVE RELEVANT KNOWLEDGE
# ==========================================

def retrieve_context(
    query,
    index,
    chunks,
    top_k=3
):
    """
    Retrieve the most relevant career
    knowledge chunks for a question.
    """

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    # Don't request more results than
    # the number of chunks available
    top_k = min(
        top_k,
        len(chunks)
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        results.append(
            {
                "text": chunks[idx]["text"],
                "source": chunks[idx]["source"],
                "score": float(score)
            }
        )

    return results


# ==========================================
# FORMAT CANDIDATE PROFILE
# ==========================================

def format_candidate_profile(profile):
    """
    Convert the parsed resume profile into
    concise text for the Career Mentor.

    Only information actually present in
    the parsed profile is included.
    """

    if not profile:

        return (
            "No resume information is available."
        )

    sections = []

    # ======================================
    # BASIC INFORMATION
    # ======================================

    name = profile.get(
        "name",
        ""
    )

    target_role = profile.get(
        "target_role",
        ""
    )

    if name:

        sections.append(
            f"Candidate Name: {name}"
        )

    if target_role:

        sections.append(
            f"Target Role: {target_role}"
        )


    # ======================================
    # SKILLS
    # ======================================

    skills = profile.get(
        "skills",
        []
    )

    skill_strings = []

    for skill in skills:

        if isinstance(skill, dict):

            value = (
                skill.get("name")
                or skill.get("skill")
                or str(skill)
            )

        else:

            value = str(skill)

        if value:

            skill_strings.append(value)

    if skill_strings:

        sections.append(
            "Skills:\n- "
            + "\n- ".join(skill_strings)
        )


    # ======================================
    # PROJECTS
    # ======================================

    projects = profile.get(
        "projects",
        []
    )

    project_strings = []

    for project in projects:

        if isinstance(project, dict):

            title = (
                project.get("title")
                or project.get("name")
                or project.get("project")
                or ""
            )

            description = project.get(
                "description",
                ""
            )

            if title and description:

                project_strings.append(
                    f"{title}: {description}"
                )

            elif title:

                project_strings.append(
                    title
                )

            elif description:

                project_strings.append(
                    description
                )

        else:

            project_strings.append(
                str(project)
            )

    if project_strings:

        sections.append(
            "Projects:\n- "
            + "\n- ".join(project_strings)
        )


    # ======================================
    # EXPERIENCE
    # ======================================

    experience = profile.get(
        "experience",
        []
    )

    experience_strings = []

    for exp in experience:

        if isinstance(exp, dict):

            role = exp.get(
                "role",
                ""
            )

            company = exp.get(
                "company",
                ""
            )

            description = exp.get(
                "description",
                ""
            )

            parts = []

            if role:

                parts.append(role)

            if company:

                parts.append(
                    f"at {company}"
                )

            if description:

                parts.append(
                    description
                )

            if parts:

                experience_strings.append(
                    " ".join(parts)
                )

        else:

            experience_strings.append(
                str(exp)
            )

    if experience_strings:

        sections.append(
            "Experience:\n- "
            + "\n- ".join(
                experience_strings
            )
        )


    # ======================================
    # EDUCATION
    # ======================================

    education = profile.get(
        "education",
        []
    )

    education_strings = []

    for edu in education:

        if isinstance(edu, dict):

            degree = edu.get(
                "degree",
                ""
            )

            institution = edu.get(
                "institution",
                ""
            )

            start_date = edu.get(
                "start_date",
                ""
            )

            end_date = edu.get(
                "end_date",
                ""
            )

            gpa = edu.get(
                "gpa",
                ""
            )

            parts = []

            if degree:

                parts.append(degree)

            if institution:

                parts.append(
                    f"at {institution}"
                )

            if start_date or end_date:

                parts.append(
                    f"({start_date} - {end_date})"
                )

            if gpa:

                parts.append(
                    f"GPA: {gpa}"
                )

            if parts:

                education_strings.append(
                    " ".join(parts)
                )

        else:

            education_strings.append(
                str(edu)
            )

    if education_strings:

        sections.append(
            "Education:\n- "
            + "\n- ".join(
                education_strings
            )
        )


    # ======================================
    # CERTIFICATIONS
    # ======================================

    certifications = profile.get(
        "certifications",
        []
    )

    certification_strings = []

    for cert in certifications:

        if isinstance(cert, dict):

            cert_name = cert.get(
                "name",
                ""
            )

            date = cert.get(
                "date",
                ""
            )

            if cert_name and date:

                certification_strings.append(
                    f"{cert_name} ({date})"
                )

            elif cert_name:

                certification_strings.append(
                    cert_name
                )

        else:

            certification_strings.append(
                str(cert)
            )

    if certification_strings:

        sections.append(
            "Certifications:\n- "
            + "\n- ".join(
                certification_strings
            )
        )


    # ======================================
    # RETURN FORMATTED PROFILE
    # ======================================

    if not sections:

        return (
            "No resume information is available."
        )

    return "\n\n".join(sections)


# ==========================================
# GENERATE CAREER MENTOR RESPONSE
# ==========================================

def ask_career_mentor(
    question,
    index,
    chunks,
    candidate_profile=None
):
    """
    Answer a career question using RAG
    with input and output guardrails.

    candidate_profile is optional so existing
    evaluation scripts continue to work.
    """

    # ======================================
    # INPUT GUARDRAIL
    # ======================================

    input_check = validate_question(
        question
    )

    if not input_check["allowed"]:

        return {
            "success": False,
            "stage": "input_guardrail",
            "answer": None,
            "reason": input_check["reason"],
            "sources": []
        }


    # ======================================
    # RETRIEVE RELEVANT KNOWLEDGE
    # ======================================

    retrieved = retrieve_context(
        question,
        index,
        chunks,
        top_k=3
    )

    if not retrieved:

        return {
            "success": False,
            "stage": "retrieval",
            "answer": None,
            "reason": (
                "No relevant career knowledge "
                "was found."
            ),
            "sources": []
        }


    # ======================================
    # BUILD RETRIEVED CONTEXT
    # ======================================

    context = "\n\n".join(
        [
            f"Source: {item['source']}\n"
            f"{item['text']}"
            for item in retrieved
        ]
    )


    # ======================================
    # BUILD CANDIDATE CONTEXT
    # ======================================

    candidate_context = (
        format_candidate_profile(
            candidate_profile
        )
    )

    has_resume = (
        candidate_profile is not None
        and bool(candidate_profile)
    )


    # ======================================
    # PERSONALIZATION INSTRUCTION
    # ======================================

    if has_resume:

        personalization_instruction = """
A resume has been uploaded.

Use the candidate profile ONLY to personalize
the response.

You may:
- Identify skills already present.
- Identify relevant skill gaps.
- Connect recommendations to projects.
- Connect recommendations to experience.
- Suggest what the candidate should learn next.

Do NOT:
- Invent skills.
- Invent experience.
- Invent projects.
- Invent certifications.
- Invent achievements.
- Claim the candidate is qualified for a role
  unless the provided information supports it.

Do not simply repeat the resume.
Use it to make the career advice more relevant.
"""

    else:

        personalization_instruction = """
No resume has been uploaded.

Give general career guidance using the
retrieved career knowledge.
"""


    # ======================================
    # GROUNDED GEMINI PROMPT
    # ======================================

    prompt = f"""
You are SmartHire GenAI's Career Mentor.

Answer the user's career question using the
retrieved career knowledge as the authoritative
source for career recommendations.

IMPORTANT RULES:

1. Use the retrieved career knowledge as your
   primary and authoritative source.

2. Do not invent career facts that are not
   supported by the retrieved knowledge.

3. Give practical and concise career advice.

4. Do not guarantee jobs, salaries,
   placements, admissions, interviews,
   promotions, or career outcomes.

5. Never claim that the candidate has a skill,
   experience, certification, project, or
   qualification unless it appears in the
   candidate profile.

6. If the retrieved career knowledge does not
   contain enough information to answer an
   important part of the question, say so
   clearly.

7. Separate:
   - Existing candidate strengths
   - Relevant skill gaps
   - Recommended next steps

8. The candidate profile is used only for
   personalization. It must not be treated as
   a replacement for the retrieved career
   knowledge.

{personalization_instruction}

==========================================
RETRIEVED CAREER KNOWLEDGE
==========================================

{context}

==========================================
CANDIDATE PROFILE
==========================================

{candidate_context}

==========================================
USER QUESTION
==========================================

{question}

==========================================
ANSWER
==========================================

Provide a helpful, concise, career-focused
answer.

If a resume is available, personalize the
answer using the candidate profile.

Do not dump the entire resume into the answer.
"""


    # ======================================
    # GENERATE RESPONSE
    # ======================================

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        answer = response.text.strip()

    except Exception as e:

        return {
            "success": False,
            "stage": "generation",
            "answer": None,
            "reason": str(e),
            "sources": retrieved
        }


    # ======================================
    # OUTPUT GUARDRAIL
    # ======================================

    # Validate against BOTH:
    # 1. Retrieved career knowledge
    # 2. Candidate resume information
    #
    # This allows personalized answers to
    # mention actual resume details without
    # being incorrectly rejected as ungrounded.

    validation_context = (
        context
        + "\n\n"
        + "Candidate Resume Information:\n"
        + candidate_context
    )

    output_check = validate_answer(
        answer,
        validation_context
    )

    if not output_check["safe"]:

        return {
            "success": False,
            "stage": "output_guardrail",
            "answer": None,
            "reason": output_check["reason"],
            "sources": retrieved
        }


    # ======================================
    # SUCCESS
    # ======================================

    return {
        "success": True,
        "stage": "complete",
        "answer": answer,
        "reason": (
            "Career Mentor response "
            "generated successfully."
        ),
        "sources": retrieved
    }