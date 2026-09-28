from app.rag_chain import (
    ask_question
)


def main():

    print()
    print("=" * 70)
    print(
        "        CPG RETAIL RAG ASSISTANT"
    )
    print("=" * 70)

    print(
        "Ask questions about products, "
        "sales, inventory, invoices,"
    )

    print(
        "promotions, returns, suppliers "
        "and store operations."
    )

    print()
    print(
        "Type 'exit' or 'quit' to stop."
    )

    print("=" * 70)

    while True:

        try:

            question = input(
                "\nYou: "
            ).strip()

            # Empty input
            if not question:

                print(
                    "Please enter a question."
                )

                continue

            # Exit
            if question.lower() in [
                "exit",
                "quit"
            ]:

                print(
                    "\nThank you for using "
                    "CPG Retail RAG Assistant!"
                )

                break

            # Ask RAG
            answer, sources = (
                ask_question(question)
            )

            print()
            print(
                "Assistant:"
            )

            print(answer)

            # Sources
            if sources:

                print()
                print(
                    "Sources:"
                )

                for source in sources:

                    print(
                        f"  - {source}"
                    )

        except KeyboardInterrupt:

            print(
                "\n\nExiting..."
            )

            break

        except Exception as error:

            print(
                f"\nError: {error}"
            )


if __name__ == "__main__":
    main()