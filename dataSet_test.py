from dataset import Dataset
import nltk
# Check if 'punkt' is already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading 'punkt' tokenizer...")
    nltk.download('punkt')
    nltk.download('punkt_tab')

# Try to load the training data
train_dataset = Dataset('data/train')

# Print how many sentences were loaded
n_sentences = sum(1 for _ in train_dataset.sentences())
print(f"Loaded {n_sentences} sentences from training data.")

# Print a few examples
for i, sentence in enumerate(train_dataset.sentences()):
    print(f"Sentence {i}:")
    for token in sentence:
        print(f"  {token['form']} ({token['start']}-{token['end']}) -> {token['tag']}")
    if i == 2:  # Only print first 3 sentences
        break