from search import search_prompt


def main() -> None:
    try:
        chain = search_prompt()
    except ValueError as error:
        print(f"Não foi possível iniciar o chat: {error}")
        return

    print("Chat iniciado. Digite 'sair' ou 'exit' para encerrar.")
    while True:
        try:
            question = input("\nPERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChat encerrado.")
            return

        if question.lower() in {"sair", "exit", "quit"}:
            print("Chat encerrado.")
            return
        if not question:
            continue

        response = chain.invoke({"question": question})
        print(f"RESPOSTA: {response}")


if __name__ == "__main__":
    main()