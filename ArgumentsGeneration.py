import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split 
import os


# Function to get embeddings for a single sentence or batch of sentences
def get_embeddings(texts):
    model_name = "nlpaueb/legal-bert-base-uncased"  # Fine-tuned for legal texts
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # Tokenize input texts
    inputs = tokenizer(texts, return_tensors="pt", padding=True,truncation=True,max_length=512 )

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token embeddings

    return embeddings

# Function to save embeddings to a file
def save_embeddings(embeddings, filename):
    np.save(filename, embeddings.cpu().numpy())  # Save embeddings as a .npy file

# Function to load embeddings from a file
def load_embeddings(filename):
    return torch.tensor(np.load(filename))  # Load embeddings as a tensor


# find n most similar cases within dataset of cases

def findSimilarCases(k, input):
    embeddingsdir = "Data/"
    #train_embeddings_file = os.path.join(embeddingsdir, 'train_embeddings.npy')
    train_embeddings_file_path = "Data/train_embeddings_600.npy"
    if os.path.exists(train_embeddings_file_path):
        print("Loading train embeddings...")
        train_embeddings = load_embeddings(train_embeddings_file_path)
    else:
        print("Calculating train embeddings...")
        train_texts = [case['facts'] for case in dataset['train']]
        train_embeddings = get_embeddings(train_texts)
        save_embeddings(train_embeddings, train_embeddings_file_path)

    # Calculate test embeddings
    #test_texts = [case['facts'] for case in dataset['test']]
    test_embeddings = get_embeddings(input)

    # Initialize cosine similarity function
    cosi = torch.nn.CosineSimilarity(dim=1)

    # Calculate cosine similarity between the single test case and each train case
    similarity_scores = cosi(test_embeddings[0].unsqueeze(0), train_embeddings)

    # Get the top k most similar cases
    top_values, top_indices = torch.topk(similarity_scores, k)
    

    # Print the top 5 most similar cases to the test case
    print("Top", k," most similar cases to the test case:")
    for i in range(min(k, len(similarity_scores))):  # Handle cases where there are fewer than 5 training examples
        index = top_indices[i].item()  # Convert tensor to scalar index
        similarity = top_values[i].item()  # Get the cosine similarity value
        print(f"Case {index}: Similarity = {similarity:.4f}")
        #print(f"Data: {dataset['train'][index]['violation']}")  # Adjust to how the original dataset stores the cases
        print("-" * 50)
    
    return top_indices.tolist()

# create dataset with the facts but only article 10 cases
def createDataSet():
    data_facts = load_dataset("RashidHaddad/ECTHR-PCR")
    data_facts = pd.DataFrame(data_facts['train'])
    data_meta = pd.read_excel('Data/data_ECHROD.xlsx')
    temp = pd.merge(data_facts, data_meta, on ='appno', how='inner')
    dataset = temp[['appno','date', 'facts', 'law','itemid', 'ccl_article=10', 'ccl_article=10-1']]
    print(type(dataset['ccl_article=10'][0]))
    dataset['violation'] = np.where((dataset['ccl_article=10'] == 1) |(dataset['ccl_article=10-1'] == 1) ,1, 0)
    dataset = dataset.drop(['ccl_article=10', 'ccl_article=10-1'], axis=1)
    data_summary = pd.read_excel('Data/data_summary2.xlsx')
    data_summary = data_summary[['appno','summary', 'summary2']]
    complete_dataset = pd.merge(dataset, data_summary, on = 'appno', how= 'inner')

    complete_dataset.to_csv('Data/article10_cases_.csv')
    complete_dataset.to_excel('Data/cases_.xlsx')
    print(complete_dataset.columns)


        

#createDataSet()

'''

data = pd.read_csv('Data/article10_cases_.csv')
train_data = data.iloc[0:600,:]
test_data = data.iloc[300:301,:]
#print(train_data)
new_case = test_data.sample()
train_data.to_csv('train.csv', index=False)
new_case.to_csv('test.csv', index = False)
new_case = pd.read_csv('test.csv')

embeddingsdir = "Data/"
     
dataset = load_dataset('csv', data_files={'train': 'train.csv',   'test': 'test.csv'})
print('new case :\n', new_case)
similar_cases = findSimilarCases(6,new_case['facts'].iloc[0] )
#print(type(similar_cases))

'''


