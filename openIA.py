from transformers import AutoTokenizer, pipeline
import torch
import warnings


warnings.filterwarnings("ignore")

#model = "meta-llama/Llama-2-7b-chat-hf" # meta-llama/Llama-2-7b-hf
model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)

llama_pipeline = pipeline(
    "text-generation",  # LLM task
    model=model,
    torch_dtype=torch.float32,
    device_map="auto",
)

def get_llama_response(prompt: str) -> None:
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        None: Prints the model's response.
    """
    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=512,
        truncation = True,
    )
    print("Chatbot:", sequences[0]['generated_text'])

while True:
    user_input = input("You: ")
    if user_input.lower() in ["bye", "quit", "exit"]:
        print("Chatbot: Goodbye!")
        break
    get_llama_response(user_input)
