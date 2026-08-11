import json
import os
import tempfile

import streamlit as st

from src.parser import extract_text, parse_resume
from src.job_matcher import JobMatcher
from src.cv_generator import generate_cv_suggestions

from src.career_rag import (
    build_career_index,
    ask_career_mentor
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SmartHire GenAI",
    page_icon="💼",
    layout="wide"
)


# ==========================================
# CACHED FUNCTIONS
# ==========================================

@st.cache_data(show_spinner="📄 Processing resume...")
def process_resume(file_bytes, file_extension):
    """
    Extract and parse a resume.

    Cached so Streamlit does not call Gemini again
    every time the page reruns for the same resume.
    """

    suffix = f".{file_extension}"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:

        text = extract_text(temp_path)
        profile = parse_resume(text)

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)

    return text, profile


@st.cache_data(show_spinner="🔎 Finding matching jobs...")
def find_matching_jobs_cached(resume_for_search):
    """
    Find matching jobs and cache the result.
    """

    matcher = JobMatcher()

    return matcher.find_matching_jobs(
        resume_for_search,
        top_k=5
    )


@st.cache_data(show_spinner="✨ Generating CV improvements...")
def generate_cv_suggestions_cached(
    resume_text,
    job_json
):
    """
    Cache CV improvement generation for the
    same resume/job combination.
    """

    job = json.loads(job_json)

    return generate_cv_suggestions(
        resume_text,
        job
    )


@st.cache_resource(
    show_spinner="🔄 Loading Career Mentor knowledge..."
)
def get_career_rag():
    """
    Build the Career Mentor FAISS index only once.

    The embedding model and vector index are
    cached between Streamlit reruns.
    """

    return build_career_index()


# ==========================================
# SESSION STATE
# ==========================================

if "career_result" not in st.session_state:
    st.session_state.career_result = None

if "career_question" not in st.session_state:
    st.session_state.career_question = ""

if "resume_key" not in st.session_state:
    st.session_state.resume_key = None

# NEW:
# Store the parsed resume profile so that the
# Career Mentor can use it even after Streamlit
# reruns the page.
if "resume_profile" not in st.session_state:
    st.session_state.resume_profile = {}


# ==========================================
# PAGE TITLE
# ==========================================

st.title("💼 SmartHire GenAI")


# ==========================================
# RESUME UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"]
)


# ==========================================
# RESUME PROCESSING
# ==========================================

if uploaded_file:

    file_extension = (
        uploaded_file.name
        .split(".")[-1]
        .lower()
    )

    file_bytes = uploaded_file.getvalue()

    current_resume_key = (
        f"{uploaded_file.name}:"
        f"{uploaded_file.size}"
    )

    # --------------------------------------
    # Detect a new resume
    # --------------------------------------

    if st.session_state.resume_key != current_resume_key:

        st.session_state.resume_key = current_resume_key

        # Clear previous Career Mentor answer
        st.session_state.career_result = None
        st.session_state.career_question = ""

        # Clear previous stored profile
        st.session_state.resume_profile = {}


    # ======================================
    # EXTRACT + PARSE RESUME
    # ======================================

    try:

        text, profile = process_resume(
            file_bytes,
            file_extension
        )

        # NEW:
        # Save parsed resume profile in session state.
        # This allows Career Mentor to access the
        # candidate profile.
        st.session_state.resume_profile = profile

        st.success(
            "✅ Resume Parsed Successfully!"
        )

    except Exception as e:

        st.error(
            "❌ Unable to parse the uploaded resume."
        )

        st.exception(e)

        st.stop()


    # ======================================
    # PREPARE RESUME FOR JOB MATCHING
    # ======================================

    skills = profile.get(
        "skills",
        []
    )

    projects = profile.get(
        "projects",
        []
    )

    experience = profile.get(
        "experience",
        []
    )


    # ======================================
    # SKILLS
    # ======================================

    skill_strings = []

    for skill in skills:

        if isinstance(skill, dict):

            skill_strings.append(
                skill.get("name")
                or skill.get("skill")
                or str(skill)
            )

        else:

            skill_strings.append(
                str(skill)
            )


    # ======================================
    # PROJECTS
    # ======================================

    project_strings = []

    for project in projects:

        if isinstance(project, dict):

            project_strings.append(
                project.get("title")
                or project.get("name")
                or project.get("project")
                or str(project)
            )

        else:

            project_strings.append(
                str(project)
            )


    # ======================================
    # EXPERIENCE
    # ======================================

    experience_strings = []

    for exp in experience:

        if isinstance(exp, dict):

            experience_strings.append(
                exp.get("role")
                or exp.get("company")
                or str(exp)
            )

        else:

            experience_strings.append(
                str(exp)
            )


    # ======================================
    # CREATE SEARCH QUERY
    # ======================================

    resume_for_search = f"""
Target Role:
{profile.get("target_role", "")}

Skills:
{", ".join(skill_strings)}

Projects:
{", ".join(project_strings)}

Experience:
{", ".join(experience_strings)}
"""


    # ======================================
    # FIND MATCHING JOBS
    # ======================================

    try:

        jobs = find_matching_jobs_cached(
            resume_for_search
        )

    except Exception as e:

        st.error(
            "❌ Unable to find matching jobs."
        )

        st.exception(e)

        jobs = []


    # ======================================
    # CANDIDATE PROFILE
    # ======================================

    st.header("👤 Candidate Profile")

    col1, col2 = st.columns(2)

    with col1:

        st.write("### Name")

        st.write(
            profile.get(
                "name",
                ""
            )
        )

        st.write("### Email")

        st.write(
            profile.get(
                "email",
                ""
            )
        )


    with col2:

        st.write("### Phone")

        st.write(
            profile.get(
                "phone",
                ""
            )
        )

        st.write("### Target Role")

        st.write(
            profile.get(
                "target_role",
                ""
            )
        )


    st.divider()


    # ======================================
    # EDUCATION
    # ======================================

    st.subheader("🎓 Education")

    education = profile.get(
        "education",
        []
    )

    if education:

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

                st.markdown(
                    f"""
### 🎓 {degree}

**Institution:** {institution}

**Duration:** {start_date} – {end_date}

**GPA:** {gpa}
"""
                )

            else:

                st.write(
                    "•",
                    str(edu)
                )

    else:

        st.info(
            "No education details found."
        )


    st.divider()


    # ======================================
    # EXPERIENCE
    # ======================================

    st.subheader("💼 Experience")

    if experience:

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

                start_date = exp.get(
                    "start_date",
                    ""
                )

                end_date = exp.get(
                    "end_date",
                    ""
                )

                description = exp.get(
                    "description",
                    ""
                )

                st.markdown(
                    f"""
### 💼 {role}

**Company:** {company}

**Duration:** {start_date} – {end_date}

{description}
"""
                )

            else:

                st.write(
                    "•",
                    str(exp)
                )

    else:

        st.info(
            "No experience found."
        )


    st.divider()


    # ======================================
    # SKILLS
    # ======================================

    st.subheader("💻 Skills")

    if skill_strings:

        cols = st.columns(3)

        for i, skill in enumerate(
            skill_strings
        ):

            cols[
                i % 3
            ].success(
                skill
            )

    else:

        st.info(
            "No skills found."
        )


    st.divider()


    # ======================================
    # PROJECTS
    # ======================================

    st.subheader("📂 Projects")

    if projects:

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

                st.markdown(
                    f"""
### 📂 {title}

{description}
"""
                )

            else:

                st.write(
                    "•",
                    str(project)
                )

    else:

        st.info(
            "No projects found."
        )


    st.divider()


    # ======================================
    # CERTIFICATIONS
    # ======================================

    st.subheader("🏆 Certifications")

    certifications = profile.get(
        "certifications",
        []
    )

    if certifications:

        for cert in certifications:

            if isinstance(cert, dict):

                name = cert.get(
                    "name",
                    ""
                )

                date = cert.get(
                    "date",
                    ""
                )

                st.markdown(
                    f"""
**🏆 {name}**

📅 {date}
"""
                )

            else:

                st.write(
                    "•",
                    str(cert)
                )

    else:

        st.info(
            "No certifications found."
        )


    st.divider()


    # ======================================
    # TOP MATCHING JOBS
    # ======================================

    st.header("🎯 Top Matching Jobs")

    if jobs:

        for i, job in enumerate(
            jobs,
            start=1
        ):

            title = job.get(
                "title",
                "Unknown Job"
            )

            match_score = job.get(
                "match_score",
                0
            )

            with st.expander(
                f"#{i} {title} • "
                f"Match Score: {match_score}%"
            ):

                # ------------------------------
                # COMPANY
                # ------------------------------

                st.write(
                    "### 🏢 Company"
                )

                st.write(
                    job.get(
                        "companyName",
                        "Not available"
                    )
                )


                # ------------------------------
                # LOCATION
                # ------------------------------

                st.write(
                    "### 📍 Location"
                )

                st.write(
                    job.get(
                        "location",
                        "Not available"
                    )
                )


                # ------------------------------
                # EXPERIENCE
                # ------------------------------

                st.write(
                    "### 💼 Experience"
                )

                st.write(
                    job.get(
                        "experience",
                        "Not specified"
                    )
                )


                # ------------------------------
                # SKILLS
                # ------------------------------

                st.write(
                    "### 🛠 Skills"
                )

                st.write(
                    job.get(
                        "tagsAndSkills",
                        "Not specified"
                    )
                )


                # ------------------------------
                # DESCRIPTION
                # ------------------------------

                st.write(
                    "### 📄 Job Description"
                )

                description = job.get(
                    "jobDescription",
                    "No description available."
                )

                st.write(
                    description[:700]
                )


                # ==================================
                # CV IMPROVEMENT GENERATOR
                # ==================================

                st.divider()

                st.subheader(
                    "✨ CV Improvement"
                )

                improve_button = st.button(
                    "✨ Improve My Resume for This Job",
                    key=f"improve_{i}"
                )

                if improve_button:

                    try:

                        job_json = json.dumps(
                            job,
                            sort_keys=True,
                            default=str
                        )

                        suggestions = (
                            generate_cv_suggestions_cached(
                                text,
                                job_json
                            )
                        )

                        st.success(
                            "✅ CV analysis generated "
                            "successfully!"
                        )


                        # ==================================
                        # MISSING SKILLS
                        # ==================================

                        st.write(
                            "### 🔴 Missing Skills"
                        )

                        missing_skills = (
                            suggestions.get(
                                "missing_skills",
                                []
                            )
                        )

                        if missing_skills:

                            for skill in missing_skills:

                                st.write(
                                    f"• {skill}"
                                )

                        else:

                            st.info(
                                "No major missing skills "
                                "identified."
                            )


                        # ==================================
                        # WEAK BULLETS
                        # ==================================

                        st.write(
                            "### 📝 Resume Bullet Improvements"
                        )

                        weak_bullets = (
                            suggestions.get(
                                "weak_bullets",
                                []
                            )
                        )

                        if weak_bullets:

                            for bullet in weak_bullets:

                                if isinstance(
                                    bullet,
                                    dict
                                ):

                                    original = (
                                        bullet.get(
                                            "original",
                                            ""
                                        )
                                    )

                                    improved = (
                                        bullet.get(
                                            "improved",
                                            ""
                                        )
                                    )

                                    st.markdown(
                                        f"""
**Original**

> {original}

**Suggested Improvement**

> {improved}
"""
                                    )

                                else:

                                    st.write(
                                        f"• {bullet}"
                                    )

                        else:

                            st.info(
                                "No weak bullets "
                                "identified."
                            )


                        # ==================================
                        # REWRITTEN SUMMARY
                        # ==================================

                        st.write(
                            "### ✍️ Rewritten "
                            "Professional Summary"
                        )

                        summary = (
                            suggestions.get(
                                "rewritten_summary",
                                ""
                            )
                        )

                        if summary:

                            st.info(
                                summary
                            )

                        else:

                            st.info(
                                "No rewritten summary "
                                "generated."
                            )


                        # ==================================
                        # OVERALL SUGGESTIONS
                        # ==================================

                        st.write(
                            "### 💡 Overall Suggestions"
                        )

                        overall_suggestions = (
                            suggestions.get(
                                "overall_suggestions",
                                []
                            )
                        )

                        if overall_suggestions:

                            for suggestion in (
                                overall_suggestions
                            ):

                                st.write(
                                    f"• {suggestion}"
                                )

                        else:

                            st.info(
                                "No additional "
                                "suggestions."
                            )


                    except Exception as e:

                        st.error(
                            "❌ Unable to generate "
                            "CV improvements."
                        )

                        st.exception(e)

    else:

        st.info(
            "No matching jobs found."
        )


# ==========================================
# AI CAREER MENTOR - RAG
# ==========================================

st.divider()

st.header(
    "🤖 AI Career Mentor"
)

st.write(
    "Ask questions about AI/ML careers, "
    "skills, projects, interviews, and "
    "career preparation."
)


# ==========================================
# RESUME-AWARE CAREER MENTOR STATUS
# ==========================================

if st.session_state.resume_profile:

    st.success(
        "📄 Resume loaded — Career Mentor will "
        "personalize advice using your profile."
    )

else:

    st.info(
        "💡 Upload a resume above to get "
        "personalized career advice."
    )


# ==========================================
# CAREER QUESTION
# ==========================================

career_question = st.text_input(
    "💬 Ask your career question",
    value=st.session_state.career_question,
    placeholder=(
        "Example: What skills should I learn "
        "to become an AI Engineer?"
    )
)


ask_mentor = st.button(
    "🤖 Ask Career Mentor"
)


# ==========================================
# PROCESS CAREER QUESTION
# ==========================================

if ask_mentor:

    st.session_state.career_question = (
        career_question
    )

    if not career_question.strip():

        st.session_state.career_result = {
            "success": False,
            "stage": "input_guardrail",
            "answer": None,
            "reason": (
                "Please enter a career question."
            ),
            "sources": []
        }

    else:

        with st.spinner(
            "🤖 Career Mentor is thinking..."
        ):

            try:

                # ----------------------------------
                # LOAD CAREER RAG
                # ----------------------------------

                career_index, career_chunks = (
                    get_career_rag()
                )


                # ----------------------------------
                # GET STORED RESUME PROFILE
                # ----------------------------------

                candidate_profile = (
                    st.session_state.resume_profile
                )


                # ----------------------------------
                # ASK CAREER MENTOR
                # ----------------------------------
                #
                # IMPORTANT:
                # The candidate profile is now passed
                # into Career Mentor.
                #
                # This allows the RAG system to combine:
                #
                # 1. Career knowledge
                # 2. Candidate's resume
                # 3. User's question
                #
                # ----------------------------------

                st.session_state.career_result = (
                    ask_career_mentor(
                        career_question,
                        career_index,
                        career_chunks,
                        candidate_profile=candidate_profile
                    )
                )


            except Exception as e:

                st.session_state.career_result = {
                    "success": False,
                    "stage": "generation",
                    "answer": None,
                    "reason": str(e),
                    "sources": []
                }


# ==========================================
# DISPLAY CAREER MENTOR RESULT
# ==========================================

career_result = (
    st.session_state.career_result
)


if career_result:

    # ======================================
    # FAILURE / GUARDRAIL
    # ======================================

    if not career_result.get(
        "success",
        False
    ):

        stage = career_result.get(
            "stage",
            "unknown"
        )

        reason = career_result.get(
            "reason",
            "Unable to answer the question."
        )


        if stage == "input_guardrail":

            st.warning(
                "🛡️ Question blocked by "
                "input guardrail."
            )

        elif stage == "output_guardrail":

            st.warning(
                "🛡️ Response blocked by "
                "output guardrail."
            )

        elif stage == "generation":

            st.error(
                "❌ Unable to generate "
                "Career Mentor response."
            )

        else:

            st.warning(
                "⚠️ Career Mentor could "
                "not answer this question."
            )


        st.info(reason)


    # ======================================
    # SUCCESSFUL RESPONSE
    # ======================================

    else:

        st.success(
            "✅ Career Mentor response "
            "generated successfully!"
        )


        # ==================================
        # MENTOR ANSWER
        # ==================================

        st.subheader(
            "💡 Career Mentor Advice"
        )

        st.write(
            career_result.get(
                "answer",
                ""
            )
        )


        # ==================================
        # KNOWLEDGE SOURCES
        # ==================================

        sources = career_result.get(
            "sources",
            []
        )

        if sources:

            st.subheader(
                "📚 Knowledge Sources"
            )

            for item in sources:

                st.markdown(
                    f"""
**Source:** `{item["source"]}`

**Relevance Score:** `{item["score"]:.3f}`
"""
                )

                with st.expander(
                    "View retrieved context"
                ):

                    st.write(
                        item["text"]
                    )

                st.divider()