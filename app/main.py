from app.generate import generate_answer

def run_cli():
    """Simple command-line loop to chat with your RAG system."""
    print("RAG system ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        result = generate_answer(question)

        print(f"\nAnswer: {result['answer']}")
        # print(f"Sources: {result['sources']}\n")


if __name__ == "__main__":
    run_cli()