from src.career_rag import (
    build_career_index,
    retrieve_context
)


print("=" * 60)
print("Building Career Mentor RAG...")
print("=" * 60)

index, chunks = build_career_index()

print(f"Knowledge chunks: {len(chunks)}")


questions = [
    "What projects should I build to become an AI Engineer?",
    "How should I prepare for an AI/ML interview?",
    "What should I learn first if I am a beginner in AI?"
]


for question in questions:

    print("\n" + "=" * 60)
    print("QUESTION:")
    print(question)
    print("=" * 60)

    results = retrieve_context(
        question,
        index,
        chunks,
        top_k=3
    )

    print("\nRETRIEVED SOURCES:")

    for result in results:

        print(
            f"\nSource: {result['source']}"
        )

        print(
            f"Score: {result['score']:.3f}"
        )

        print(
            f"Content: {result['text'][:500]}..."
        )