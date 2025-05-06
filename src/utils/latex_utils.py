import string
import json

def tokenize_latex(latex):
    """Tokenize LaTeX formula into a list of tokens."""
    latex = latex.strip()
    latex = latex.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = latex.split()  # Tokenize by spaces
    return tokens

def build_vocab(labels, min_freq=1):
    """Build vocabulary from a list of LaTeX labels."""
    vocab = {}
    for label in labels:
        tokens = tokenize_latex(label)
        for token in tokens:
            vocab[token] = vocab.get(token, 0) + 1
    
    # Remove tokens that don't meet frequency threshold
    vocab = {word: count for word, count in vocab.items() if count >= min_freq}
    
    # Add special tokens
    vocab['<PAD>'] = 0
    vocab['<START>'] = len(vocab)
    vocab['<END>'] = len(vocab) + 1
    vocab['<UNK>'] = len(vocab) + 2  # Unknown token
    
    return vocab

def save_vocab(vocab, vocab_file='vocab.json'):
    """Save vocabulary to a JSON file."""
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f)
    print(f"Vocabulary saved to {vocab_file}")

def load_vocab(vocab_file='vocab.json'):
    """Load vocabulary from a JSON file."""
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    return vocab
