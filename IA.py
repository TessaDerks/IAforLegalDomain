from sympy import symbols, And, Or, Not, Implies
from transformers import pipeline
from collections import defaultdict
import ArgumentsGeneration


# Step 1: Define the symbolic reasoning system for Article 10 (Freedom of Speech)
# We'll use basic propositional logic for legal reasoning.
FreedomOfSpeech, JustifiedRestriction, Violation, Compliance = symbols("FreedomOfSpeech JustifiedRestriction Violation Compliance")
knowledge_base = [
    Implies(FreedomOfSpeech, Compliance),  # If Freedom of Speech is upheld, there is compliance
    Implies(Violation, Not(Compliance)),  # If there's a violation, there's no compliance
    Implies(Not(JustifiedRestriction), Violation)  # If a restriction isn't justified, it's a violation
]

# Step 2: Initialize the legal-trained LLM (e.g., using Hugging Face transformers)
# use pretrained bert model (pretrained with ECHR data) from https://huggingface.co/nlpaueb/bert-base-uncased-echr
llm = pipeline("text-generation", model="tiiuae/falcon-7b-instruct")

# ECHR Article 10 database (simplified for demonstration purposes)
echr_articles = {
    "Article 10": "Everyone has the right to freedom of expression. This right shall include freedom to hold opinions and to receive and impart information and ideas without interference by public authority and regardless of frontiers."
}

# Step 3: Argumentation Framework (Prakken and Sartor inspired)
class Argument:
    def __init__(self, conclusion, premises, support):
        self.conclusion = conclusion
        self.premises = premises
        self.support = support
        self.attacks = []  # Tracks arguments attacking this one

    def add_attack(self, attacker):
        self.attacks.append(attacker)

    def is_defeated(self):
        return any(attacker.is_stronger_than(self) for attacker in self.attacks)

    def is_stronger_than(self, other):
        # Placeholder: Define strength of arguments (e.g., based on priority, evidence)
        return len(self.support) > len(other.support)

# Argument graph
arguments = []

def construct_argument(conclusion, premises, support):
    argument = Argument(conclusion, premises, support)
    arguments.append(argument)
    return argument

def check_argument_acceptability(argument):
    """Check if an argument is acceptable given the attacks and their strength."""
    if argument.is_defeated():
        return False
    return True

def add_attack(attacker, target):
    """Record an attack from one argument to another."""
    target.add_attack(attacker)

# Step 4: Helper functions
def interpret_input(user_input):
    """Use the LLM to interpret user input and generate symbolic queries."""
    prompt = (
        f"You are a legal expert. Convert the following statement into a logical query "
        f"based on ECHR Article 10: '{user_input}'"
    )
    response = llm(prompt, max_length=512, num_return_sequences=1)
    query = response[0]['generated_text']
    return query.strip()

def explain_article(article_number):
    """Fetch and return the explanation for a specific ECHR article."""
    return echr_articles.get(article_number, "Article not found in the database.")

# Main loop
if __name__ == "__main__":
    print("Welcome to the ECHR Freedom of Speech (Article 10) Legal Reasoning System!")
    
    while True:
        user_input = input("Enter the case you want to discuss 'explain Article 10' (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break

        # Step 1: Use LLM to interpret the input
        #logical_query = interpret_input(user_input)
        #print(f"Logical Query: {logical_query}")

        # Step 2: Construct an argument
        #support = ["ECHR Article 10"]  # Example: could include evidence or precedence
        #premises = [logical_query]  # Simplified for demonstration
        #argument = construct_argument(logical_query, premises, support)
        #print(f"Constructed Argument: {argument.conclusion} with premises {argument.premises}")

        # Step 3: Evaluate argument acceptability
        #acceptability = check_argument_acceptability(argument)
        #status = "acceptable" if acceptability else "defeated"
        #print(f"Argument Status: {status}")

        # Optional: Add counterarguments
        counter_input = input("Would you like to add a counterargument? (yes/no): ").strip().lower()
        if counter_input == "yes":
            counter_argument_text = input("Enter the counterargument: ")
            counter_query = interpret_input(counter_argument_text)
            counter_argument = construct_argument(counter_query, [counter_query], ["User-provided counter"])
            add_attack(counter_argument, argument)
            print(f"Counterargument added: {counter_argument.conclusion}")

            # Reevaluate acceptability
            acceptability = check_argument_acceptability(argument)
            status = "acceptable" if acceptability else "defeated"
            print(f"Updated Argument Status: {status}")

        
