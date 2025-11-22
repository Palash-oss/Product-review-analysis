"""
Text preprocessing and tokenization utilities.
Includes vocabulary building and text cleaning.
"""

import re
import string
from typing import List, Dict, Tuple
import numpy as np
from collections import Counter
import pickle


class TextPreprocessor:
    """Advanced text preprocessing for sentiment analysis."""
    
    def __init__(self, max_vocab_size: int = 50000, max_seq_length: int = 128):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.word_freq: Counter = Counter()
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.START_TOKEN, self.END_TOKEN]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags (keep the text part)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Clean text first
        text = self.clean_text(text)
        
        # Simple word tokenization (can be replaced with more sophisticated tokenizers)
        # Keep some punctuation for sentiment (!, ?, etc.)
        tokens = re.findall(r'\b\w+\b|[!?.,]', text)
        
        return tokens
    
    def build_vocabulary(self, texts: List[str]):
        """Build vocabulary from a list of texts."""
        # Count word frequencies
        for text in texts:
            tokens = self.tokenize(text)
            self.word_freq.update(tokens)
        
        # Create vocabulary with most common words
        most_common = self.word_freq.most_common(self.max_vocab_size - len(self.special_tokens))
        
        # Add special tokens first
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        
        # Add most common words
        for idx, (word, _) in enumerate(most_common, start=len(self.special_tokens)):
            self.vocab[word] = idx
        
        # Create reverse vocabulary
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text to sequence of token IDs."""
        tokens = self.tokenize(text)
        
        # Add special tokens if requested
        if add_special_tokens:
            tokens = [self.START_TOKEN] + tokens + [self.END_TOKEN]
        
        # Convert to IDs
        token_ids = [
            self.vocab.get(token, self.vocab[self.UNK_TOKEN])
            for token in tokens
        ]
        
        # Truncate or pad to max_seq_length
        if len(token_ids) > self.max_seq_length:
            token_ids = token_ids[:self.max_seq_length]
        else:
            token_ids = token_ids + [self.vocab[self.PAD_TOKEN]] * (self.max_seq_length - len(token_ids))
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert sequence of token IDs back to text."""
        tokens = [
            self.reverse_vocab.get(idx, self.UNK_TOKEN)
            for idx in token_ids
            if idx != self.vocab[self.PAD_TOKEN]
        ]
        
        # Remove special tokens
        tokens = [t for t in tokens if t not in self.special_tokens]
        
        return ' '.join(tokens)
    
    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts."""
        return np.array([self.encode(text) for text in texts])
    
    def get_attention_mask(self, token_ids: List[int]) -> List[int]:
        """Generate attention mask (1 for real tokens, 0 for padding)."""
        return [1 if idx != self.vocab[self.PAD_TOKEN] else 0 for idx in token_ids]
    
    def save(self, path: str):
        """Save preprocessor state."""
        state = {
            'vocab': self.vocab,
            'reverse_vocab': self.reverse_vocab,
            'word_freq': self.word_freq,
            'max_vocab_size': self.max_vocab_size,
            'max_seq_length': self.max_seq_length
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str):
        """Load preprocessor state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.vocab = state['vocab']
        self.reverse_vocab = state['reverse_vocab']
        self.word_freq = state['word_freq']
        self.max_vocab_size = state['max_vocab_size']
        self.max_seq_length = state['max_seq_length']


def load_glove_embeddings(glove_path: str, vocab: Dict[str, int], embedding_dim: int = 300) -> np.ndarray:
    """
    Load pretrained GloVe embeddings for the vocabulary.
    
    Args:
        glove_path: Path to GloVe embeddings file
        vocab: Vocabulary dictionary
        embedding_dim: Dimension of embeddings
    
    Returns:
        Embedding matrix of shape [vocab_size, embedding_dim]
    """
    embeddings = np.random.randn(len(vocab), embedding_dim) * 0.01
    
    # Load GloVe
    found = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in vocab:
                vector = np.array(values[1:], dtype='float32')
                embeddings[vocab[word]] = vector
                found += 1
    
    print(f"Loaded {found}/{len(vocab)} word vectors from GloVe")
    
    # Set padding embedding to zeros
    if '<PAD>' in vocab:
        embeddings[vocab['<PAD>']] = np.zeros(embedding_dim)
    
    return embeddings
