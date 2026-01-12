import json
import re
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download standard stopword list
nltk.download('stopwords', quiet=True)

def visualize_clean_words(json_path):
    # 1. Load Data
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return print(f"Error loading file: {e}")

    # 2. Extract all text recursively
    all_text = []
    def extract_strings(obj):
        if isinstance(obj, str): all_text.append(obj)
        elif isinstance(obj, dict): 
            for v in obj.values(): extract_strings(v)
        elif isinstance(obj, list):
            for i in obj: extract_strings(i)
    extract_strings(data)

    # 3. Normalize: Lowercase -> Remove Punctuation -> Tokenize
    full_text = " ".join(all_text).lower()
    # Regex: Replace anything that is NOT a lowercase letter or space with empty string
    clean_text = re.sub(r'[^a-z\s]', '', full_text)
    words = clean_text.split()

    # 4. Filter Stop Words
    stop_set = set(stopwords.words('english'))
    # Filter stopwords and accidental empty strings or single letters
    filtered_words = [w for w in words if w not in stop_set and len(w) > 1]

    # 5. Visualize Top 20
    counts = Counter(filtered_words).most_common(20)
    print(Counter(filtered_words)["blank"])
    if not counts: return print("No words found after filtering.")

    labels, values = zip(*counts)

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color='teal')
    plt.title(f'Top 20 Common Words in {json_path}')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_clean_words('/Users/USER/Desktop/University/Semester 7/CS Adverserial ML/Project/json_results/emotion_steering_results_blank.json')