from transformers import pipeline

classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')


def classify_text(description):

    # Define labels for classification (accessible, not accessible)
    labels = ["accessible", "not accessible"]

    try:
        result = classifier(description, candidate_labels=labels)
        return result
    except Exception as e:
        print(f"Error in text classification: {e}")
        raise
