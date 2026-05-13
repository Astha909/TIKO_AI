from inference.generate_lstm import generate_response
print("TIKO Chatbot Started")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot stopped")
        break

    response = generate_response(user_input)
    print("Bot:", response)