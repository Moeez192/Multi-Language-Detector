N-Gram Based Language IdentifierThis repository contains a from-scratch Python implementation of a classic, n-gram-based language identification method.It works by creating "profiles" of the most frequent character n-grams for a set of known languages and then comparing an unknown text's profile against them to find the closest match.How it WorksThe logic is based on the 1994 paper by Cavnar and Trenkle.Profile Creation (Training):For each language, the fit() method ingests a large sample text.It cleans the text (lowercase, remove punctuation) and generates all character n-grams (e.g., from n=1 to n=5).It counts the frequencies of all n-grams and saves a ranked list of the top_n (e.g., top 300) most frequent ones. This list is the language's "profile" or "fingerprint."Language Detection (Prediction):When the predict() method receives a new text string, it generates its own top_n n-gram profile in the same way.It then calculates a "distance" score between the new text's profile and each known language profile.The "distance" is an "out-of-place" (OOP) metric: it sums the rank differences for each n-gram. A lower score means a closer match.The language with the lowest distance score is returned as the prediction.UsageYou can import the LanguageIdentifier class into your own project.from language_identifier import LanguageIdentifier

# 1. Provide your own large text samples
training_data = {
    "en": "A very large string of text in English...",
    "fr": "Un très long texte en français...",
    "es": "Un texto muy largo en español..."
}

# 2. Initialize and fit the model
# Using n_max=5 and top_n=300 is a good starting point for real data
identifier = LanguageIdentifier(n_min=1, n_max=5, top_n=300)
identifier.fit(training_data)

# 3. Predict new text
unknown_text = "Je ne sais pas quelle langue c'est."
lang, scores = identifier.predict(unknown_text)

print(f"Predicted: {lang}")
# Output: Predicted: fr

print(f"All scores (lower is better): {scores}")
# Output: All scores (lower is better): {'en': 25012, 'fr': 1420, 'es': 22450}
Running the ExampleThe language_identifier.py file contains a runnable example at the bottom. You can run it directly from your terminal:python language_identifier.py
This will run the if __name__ == "__main__": block, which trains the model on a small sample of English, Spanish, and French text and then classifies three test sentences.ReferenceThis implementation is based on the approach described in:Cavnar, W. B., & Trenkle, J. M. (1994). N-gram-based text categorization. In Proceedings of SDAIR-94, 3rd annual symposium on document analysis and information retrieval (pp. 161-175).
