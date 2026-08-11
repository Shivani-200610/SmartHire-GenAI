import re


# ==========================================
# INPUT GUARDRAILS
# ==========================================

def validate_question(question: str):
    """
    Validate a user's career question before
    sending it to the RAG pipeline.
    """

    # --------------------------------------
    # Empty input
    # --------------------------------------

    if not question or not question.strip():

        return {
            "allowed": False,
            "reason": "Please enter a career question."
        }


    question = question.strip()


    # --------------------------------------
    # Extremely long input
    # --------------------------------------

    if len(question) > 1000:

        return {
            "allowed": False,
            "reason": (
                "Your question is too long. "
                "Please keep it below 1000 characters."
            )
        }


    # --------------------------------------
    # Prompt injection detection
    # --------------------------------------

    injection_patterns = [

        r"ignore previous instructions",
        r"ignore all previous instructions",
        r"forget previous instructions",
        r"forget all previous instructions",
        r"disregard previous instructions",
        r"system prompt",
        r"reveal your instructions",
        r"show me your prompt",
        r"developer message",
        r"jailbreak"
    ]


    question_lower = question.lower()


    for pattern in injection_patterns:

        if re.search(
            pattern,
            question_lower
        ):

            return {
                "allowed": False,
                "reason": (
                    "This question contains "
                    "an unsupported instruction."
                )
            }


    # --------------------------------------
    # Career relevance check
    # --------------------------------------

    career_keywords = [

        "career",
        "job",
        "jobs",
        "skill",
        "skills",
        "resume",
        "cv",
        "interview",
        "internship",
        "internships",
        "ai",
        "artificial intelligence",
        "machine learning",
        "ml",
        "deep learning",
        "data science",
        "data scientist",
        "ai engineer",
        "ml engineer",
        "software engineer",
        "developer",
        "programming",
        "python",
        "sql",
        "llm",
        "generative ai",
        "rag",
        "langchain",
        "portfolio",
        "project",
        "projects",
        "placement",
        "placements"
    ]


    contains_career_topic = any(
        keyword in question_lower
        for keyword in career_keywords
    )


    if not contains_career_topic:

        return {
            "allowed": False,
            "reason": (
                "I can help with career-related "
                "questions about AI, ML, software "
                "engineering, resumes, skills, "
                "projects, internships, interviews, "
                "and placements."
            )
        }


    # --------------------------------------
    # Question is valid
    # --------------------------------------

    return {
        "allowed": True,
        "reason": "Question passed input guardrails."
    }


# ==========================================
# OUTPUT GUARDRAILS
# ==========================================

def validate_answer(
    answer: str,
    retrieved_context: str
):
    """
    Perform basic validation on the generated
    Career Mentor response.

    This checks whether the answer exists and
    whether it appears grounded in the retrieved
    career knowledge.
    """

    # --------------------------------------
    # Empty answer
    # --------------------------------------

    if not answer or not answer.strip():

        return {
            "safe": False,
            "reason": "The mentor generated an empty response."
        }


    answer = answer.strip()


    # --------------------------------------
    # Extremely long response
    # --------------------------------------

    if len(answer) > 5000:

        return {
            "safe": False,
            "reason": (
                "The generated response is "
                "too long."
            )
        }


    # --------------------------------------
    # Check for unsupported certainty
    # --------------------------------------

    risky_phrases = [

        "you are guaranteed to get a job",
        "guaranteed job",
        "100% guaranteed",
        "you will definitely get hired",
        "guaranteed placement",
        "you will definitely get placed"
    ]


    answer_lower = answer.lower()


    for phrase in risky_phrases:

        if phrase in answer_lower:

            return {
                "safe": False,
                "reason": (
                    "The response contains an "
                    "unsupported guarantee."
                )
            }


    # --------------------------------------
    # Basic grounding check
    # --------------------------------------

    context_words = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            retrieved_context.lower()
        )
    )


    answer_words = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            answer_lower
        )
    )


    if context_words:

        overlap = (
            len(
                answer_words.intersection(
                    context_words
                )
            )
            / len(answer_words)
        )

    else:

        overlap = 0


    # We don't require 100% overlap because
    # Gemini may naturally rephrase the context.

    if overlap < 0.05:

        return {
            "safe": False,
            "reason": (
                "The response does not appear "
                "sufficiently grounded in the "
                "retrieved career knowledge."
            )
        }


    # --------------------------------------
    # Output passed
    # --------------------------------------

    return {
        "safe": True,
        "reason": "Response passed output guardrails."
    }