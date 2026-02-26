import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel

# Check if CUDA (GPU) is available for faster processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def cls_pooling(model_output):
    """
    Performs CLS Pooling on the model output.
    
    The 'CLS' token is the first token in the sequence of the BERT-family models.
    It is often used to represent the entire sequence (sentence/document) embedding.
    
    Args:
        model_output: The raw output from the Transformer model.
        
    Returns:
        torch.Tensor: The embedding vector for the [CLS] token.
    """
    # model_output.last_hidden_state has shape (batch_size, sequence_length, hidden_size)
    # We take all items in the batch, and the 0-th token (CLS) from the sequence.
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list, tokenizer, model):
    """
    Generates semantic embeddings for a list of texts using a pre-trained Transformer model.
    
    Args:
        text_list (list): List of strings to embed.
        tokenizer: The Hugging Face tokenizer.
        model: The Hugging Face model.
        
    Returns:
        torch.Tensor: A tensor of embeddings.
    """
    # Tokenize the input text. 
    # padding=True: pads shorter sequences to the longest in the batch.
    # truncation=True: cuts off sequences longer than the model's max length.
    # return_tensors="pt": returns PyTorch tensors.
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    
    # Move the encoded inputs to the GPU (if available)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    # Generate embeddings (forward pass)
    # no_grad() prevents calculating gradients, saving memory and computation since we are not training.
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Extract the sentence embedding using CLS pooling
    return cls_pooling(model_output)

def main():
    # ---------------------------------------------------------
    # 1. Load and Preprocess Data
    # ---------------------------------------------------------
    print("Loading dataset 'talkmap/banking-conversation-corpus'...")
    dataset = load_dataset("talkmap/banking-conversation-corpus", split="train")
    
    # Filter out short or empty messages to ensure quality
    print("Filtering dataset...")
    dataset = dataset.filter(lambda x: x['text'] is not None and len(x['text']) > 10)
    
    # Limit to a sample for demonstration purposes (faster execution)
    sample_size = 1000
    print(f"Selecting a sample of {sample_size} records...")
    dataset_sample = dataset.select(range(sample_size))
    
    # Convert to Pandas for easier manipulation
    df = pd.DataFrame(dataset_sample)
    
    # Sort by conversation ID and timestamp to reconstruct the dialogue flow
    df = df.sort_values(by=['conversation_id', 'date_time'])
    
    # Format the conversation text: "SPEAKER: message"
    df['formatted_text'] = df['speaker'].str.upper() + ": " + df['text'].fillna('')
    
    # Group messages by conversation_id to form the full dialogue transcript
    print("Grouping conversations...")
    df_conversations = (
        df.groupby('conversation_id', sort=False)['formatted_text']
        .apply(lambda x: "\n".join(x))
        .reset_index(name='full_conversation')
    )
    
    # Convert back to Hugging Face Dataset format for compatibility with their tools
    processed_dataset = Dataset.from_pandas(df_conversations)
    print(f"Processed {len(processed_dataset)} conversations.")

    # ---------------------------------------------------------
    # 2. Load the Embedding Model
    # ---------------------------------------------------------
    # We use 'multi-qa-mpnet-base-dot-v1', a model optimized for semantic search.
    # It maps sentences to a 768-dimensional dense vector space.
    model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    print(f"Loading model: {model_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)
    model.to(device)

    # ---------------------------------------------------------
    # 3. Generate Embeddings & Index
    # ---------------------------------------------------------
    print("Generating embeddings for all conversations (this may take a moment)...")
    # We map the 'get_embeddings' function over the dataset.
    # The result is a new column 'embeddings' containing the vector for each conversation.
    embeddings_dataset = processed_dataset.map(
        lambda x: {"embeddings": get_embeddings(x["full_conversation"], tokenizer, model).detach().cpu().numpy()[0]}
    )
    
    # FAISS (Facebook AI Similarity Search) is a library for efficient similarity search of dense vectors.
    # We add an index to the 'embeddings' column to enable fast nearest-neighbor search.
    print("Building FAISS index...")
    embeddings_dataset.add_faiss_index(column="embeddings")

    # ---------------------------------------------------------
    # 4. Perform a Semantic Search
    # ---------------------------------------------------------
    query = "I lost my card and need to block it"
    print(f"\nPerforming semantic search for query: '{query}'")
    
    # Embed the query using the same model
    query_embedding = get_embeddings([query], tokenizer, model).cpu().detach().numpy()
    
    # Retrieve the 3 nearest neighbors (most similar conversations)
    scores, samples = embeddings_dataset.get_nearest_examples(
        "embeddings", query_embedding, k=3
    )
    
    # Display results
    print("\n--- Top Matching Conversations ---")
    for i, score in enumerate(scores):
        print(f"\nResult {i+1} (Score: {score:.4f}):")
        # Truncate output for readability
        print(samples['full_conversation'][i][:500] + "...") 

if __name__ == "__main__":
    main()
