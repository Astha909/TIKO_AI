from inference.generate_lstm import generate_response
print("TIKO Chatbot Started")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot stopped")
        break

    mode_input = user_input

    if user_input.startswith("funny"):
        mode_input = "funny " + user_input

    elif user_input.startswith("sarcastic"):
        mode_input = "sarcastic " + user_input

    elif user_input.startswith("friendly"):
        mode_input = "friendly " + user_input

    response = generate_response(mode_input)
    print("Bot:", response)