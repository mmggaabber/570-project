## Final Project for Phys 570 - Machine Learning for Science MCQs

This is the code implementation of my final project for **Phys 570** at **Purdue University**. In this project, I use machine learning techniques to solve **physics and science multiple-choice questions** (MCQs). The project involves several components, including baseline models, BERT fine-tuning, and retrieval-augmented generation (RAG).

You can access the full walkthrough of this project in the following presentation:
[Project Walkthrough](https://drive.google.com/file/d/1KBg6pgkWvVlH1pGPQY25nZis4VJfcQnp/view?usp=sharing)

### Dataset
The dataset used for training and testing the models is available here:
[Download Dataset](https://drive.google.com/file/d/1veQDqHIz3M7RYvw7XQQdk_m1HafHHZI1/view?usp=sharing)

### Code
The full code is included in the .py file in this repo. Below, I will just highlight the main parts of the code so that you can use a similar approach and apply to your problem.

### Project Components
This project is divided into several parts, each tackling a different approach to solving the MCQs.

## 1. Baseline Model Using External API (Flan-T5 from Huggingface)
For the baseline model, I used **Flan-T5**, an API from **Huggingface**. The model is called to solve each question by providing it with the question and possible answers. 

### Example code to use Flan-T5 API:
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Initialize the T5 tokenizer and model
flan_t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
flan_t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

def solve_question_t5(question, choices):
    input_text = f"Question: {question} Choices: {choices}"
    input_ids = flan_t5_tokenizer(input_text, return_tensors="pt").input_ids
    outputs = flan_t5_model.generate(input_ids)
    answer = flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
```

## 2. BERT Model with Softmax Function

I used **BERT** (`bert-base-uncased`) to predict the correct answer for each MCQ. The model takes the question and the choices as input, and I used a **softmax** function to determine the most likely answer.

### Example code to use BERT model:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize BERT tokenizer and model
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

def preprocess_data(data, tokenizer):
    inputs, labels = [], []
    for _, row in data.iterrows():
        prompt = row['prompt']
        choices = "\n".join([f"{opt}: {row[opt]}" for opt in ['A', 'B', 'C', 'D', 'E']])
        inputs.append(f"Question: {prompt} Choices: {choices}")
        labels.append(ord(row['correct_answer']) - ord('A'))
    tokenized_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    return tokenized_inputs, torch.tensor(labels)
```

## 3. Fine-Tuning BERT with Hyperparameters

To optimize BERT's performance for MCQ question answering, I fine-tuned the model by adjusting several hyperparameters, including:

- **Learning Rate**
- **Epochs**
- **Batch Size**
- **Sequence Length**
- **Number of Multi-Head Attention Layers**

### Hyperparameters Setup:

```python
from transformers import Trainer, TrainingArguments

# Hyperparameter values
learning_rate = 2e-5
batch_size = 16
epochs = 3
sequence_length = 512

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    max_length=sequence_length
)

# Trainer Setup
trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=bert_tokenizer
)

trainer.train()
```

## 4. Retrieval-Augmented Generation (RAG) System

To enhance the performance of the BERT model and improve the accuracy of the MCQ question answering, I implemented a Retrieval-Augmented Generation (RAG) system. This system leverages context-based dataset creation by embedding the question and answer choices into a vector space using Cohere's embeddings, and then utilizing FAISS for efficient similarity search.

### Steps for RAG Implementation:

1. **Cohere Embeddings**:
   - First, I used Cohere's pre-trained embeddings to represent the text (questions and answer choices) as high-dimensional vectors. This enables the model to retrieve similar content based on semantic meaning.
   - The cohere's embeddings of WIKI articles are available here:
https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings?ref=cohere-ai.ghost.io

```python
import cohere
from cohere.embeddings import EmbeddingModel

# Initialize Cohere API client
co = cohere.Client('YOUR_COHERE_API_KEY')

# Get embeddings for the MCQ dataset
def get_embeddings(texts):
    embeddings = co.embed(texts=texts).embeddings
    return embeddings

# Sample MCQ questions and answers
texts = [
    "What is the capital of France?",
    "A. Berlin",
    "B. Madrid",
    "C. Paris",
    "D. Rome"
]

embeddings = get_embeddings(texts)
```

2. **Faiss**:

After generating embeddings, I indexed them using FAISS to enable fast similarity search. FAISS helps retrieve the most relevant question-answer pairs based on the question's embedding.
```python
import faiss
import numpy as np

# Convert embeddings to a NumPy array for FAISS
embeddings_np = np.array(embeddings).astype('float32')

# Create FAISS index
index = faiss.IndexFlatL2(embeddings_np.shape[1])  # Use L2 distance for similarity

# Add embeddings to FAISS index
index.add(embeddings_np)

```
2. **Context-based Dataset Creation:**:

```python
# Query with a new question
query = "What is the capital of Italy?"

# Get the embedding for the query
query_embedding = get_embeddings([query])[0]

# Search for the nearest neighbors
D, I = index.search(np.array([query_embedding]).astype('float32'), k=3)

# Retrieve the most similar MCQ options
context = [texts[i] for i in I[0]]
print("Retrieved context:", context)
```

For the full code, please see the .py file in this repo. 

If you have any questions, please reach out to gaber@purdue.edu.





