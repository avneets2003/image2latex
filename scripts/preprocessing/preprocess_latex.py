import string
import json

def tokenize_latex(latex):
    """Tokenize LaTeX formula into a list of tokens."""
    latex = latex.strip()
    latex = latex.translate(str.maketrans('', '', string.punctuation))
    return latex.split()

def build_vocab(labels):
    """Build vocabulary from list of LaTeX labels."""
    vocab = set()
    for label in labels:
        tokens = tokenize_latex(label)
        vocab.update(tokens)
    vocab = {word: idx for idx, word in enumerate(vocab, start=1)}
    vocab['<PAD>'] = 0  # Padding token
    vocab['<START>'] = len(vocab)  # Start token
    vocab['<END>'] = len(vocab) + 1  # End token
    return vocab

if __name__ == "__main__":
    train_file = "data/sample/train.lst"
    
    with open(train_file, 'r') as f:
        labels = [line.strip().split()[1] for line in f]  # Extract LaTeX labels
        
    vocab = build_vocab(labels)
    
    with open("vocab.json", 'w') as f:
        json.dump(vocab, f)
