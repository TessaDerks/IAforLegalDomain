from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the tokenizer and chat-tuned model
model_name = "tiiuae/falcon-7b-instruct"  # Adjust model size (7b, 13b, 70b) based on hardware
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",  # Automatically uses GPU if available
    load_in_8bit=True   # Optional for low-resource systems, True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_response(prompt, history=[]):
    # Combine history with the current prompt
    conversation = "".join([f"User: {u}\nAssistant: {a}\n" for u, a in history])
    conversation += f"User: {prompt}\nAssistant:"

    # Tokenize the input
    inputs = tokenizer(conversation, return_tensors="pt").to(device)
    # Generate a response
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.2,  # Adjust for randomness
        top_p=0.5        # Adjust for diversity
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the Assistant's reply (after "Assistant:")
    reply = response.split("Assistant:")[-1].strip()
    return reply

if __name__ == "__main__":
    print("LLaMA 2 Chat Assistant (Interactive Mode)")
    print("Type 'exit' to end the conversation.\n")

    # Conversation history
    history = []

    while True:
        # Get user input
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Generate model's response
        response = generate_response(user_input, history)

        # Display the response
        print(f"Assistant: {response}")

        # Update conversation history
        history.append((user_input, response))

