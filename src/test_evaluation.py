from src.career_rag import (
    build_career_index,
    retrieve_context
)

from src.evaluation import (
    evaluate_retrieval,
    calculate_summary
)


# ==========================================
# BUILD RAG INDEX
# ==========================================

print("=" * 60)
print("SMART HIRE EVALUATION")
print("=" * 60)

print("\nBuilding Career RAG index...")

index, chunks = build_career_index()

print(
    f"Knowledge chunks: {len(chunks)}"
)


# ==========================================
# RUN EVALUATION
# ==========================================

print("\nRunning evaluation...")

results = evaluate_retrieval(
    index,
    chunks
)


# ==========================================
# DISPLAY INDIVIDUAL RESULTS
# ==========================================

for i, result in enumerate(
    results,
    start=1
):

    print("\n" + "-" * 60)

    print(
        f"Test #{i}"
    )

    print(
        f"Question: {result['question']}"
    )

    print(
        f"Type: {result['type']}"
    )

    print(
        f"Score: {result['score']:.2f}"
    )

    print(
        f"Passed: {result['passed']}"
    )

    print(
        f"Reason: {result['reason']}"
    )


# ==========================================
# DEBUG TEST #1 RETRIEVAL
# ==========================================

question = (
    "What skills should I learn "
    "to become an AI Engineer?"
)

print("\n" + "=" * 60)
print("DEBUG: RETRIEVED CONTEXT FOR TEST #1")
print("=" * 60)

retrieved = retrieve_context(
    question,
    index,
    chunks,
    top_k=3
)

for i, item in enumerate(
    retrieved,
    start=1
):

    print("\n" + "-" * 60)

    print(
        f"Result #{i}"
    )

    print(
        f"Source: {item['source']}"
    )

    print(
        f"Score: {item['score']:.3f}"
    )

    print("\nContent:")

    print(
        item["text"]
    )


# ==========================================
# SUMMARY
# ==========================================

summary = calculate_summary(
    results
)

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)

print(
    f"Total Tests     : {summary['total']}"
)

print(
    f"Passed          : {summary['passed']}"
)

print(
    f"Failed          : {summary['failed']}"
)

print(
    f"Pass Rate       : "
    f"{summary['pass_rate'] * 100:.2f}%"
)

print("=" * 60)