from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define accessibility-related queries
accessibility_queries = [
    "ramp", "curb ramp", "flat walkway", "step-free entrance", "threshold-free entrance", "zero-step entry", "wide door", 
    "automatic door", "lever-style door handles", "sliding doors", 
    "flat surface", "wide hallway", "sufficient space for wheelchair turn", 
    "accessible entrance", "wheelchair accessible", "turning space for wheelchair", "accessible parking",
    "elevator", "stairs", "handrail", 
    "accessible bathroom", "accessible toilet", "raised toilet seat", "grab bars", "grab bars in shower", 
    "roll-in shower", "wide shower entrance", "adjustable shower head", 
    "non-slip flooring", 
    "accessible sink", "knee clearance under sink", "lowered shelves", 
    "accessible kitchen", "accessible kitchen island", "accessible appliances", "accessible refrigerator", 
    "low counters", "adjustable-height counters", 
    "accessible vanity", "accessible closet", "accessible indoor space",
    "accessible bedroom", "accessible light switches", "accessible power outlets", 
    "accessible thermostat", "automatic light sensors", 
    "accessible emergency exit", "accessible intercom system", 
    "accessible patio", "accessible outdoor space", "accessible garden", 
    "emergency alarm system with visual and sound signals", 
    "adjustable-height desks", "accessible window openers"
]

# Define room type queries
room_type_queries = [
    "bathroom", "kitchen", "living room", "bedroom", 
    "office", "hallway", "corridor",  "dining room", 
    "balcony", "terrace"
]

# Function to classify image and check for accessibility features
def classify_image(image_file, confidence_threshold=0.5):
    try:
        # Load image
        image = Image.open(image_file)

        # Process image and text (queries)
        inputs = processor(text=accessibility_queries + room_type_queries, images=image, return_tensors="pt", padding=True)

        # Forward pass through the model
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)  # Get probabilities for each query

        # Collect results (accessibility feature, confidence score)
        results = []
        best_guess_feature = None
        highest_confidence = 0

        for i, query in enumerate(accessibility_queries + room_type_queries):
            confidence_score = probs[0, i].item()

            # Track the highest confidence feature
            if confidence_score > highest_confidence:
                highest_confidence = confidence_score
                best_guess_feature = {"feature": query, "score": round(confidence_score, 3)}

            # Add feature if it passes the threshold
            if confidence_score >= confidence_threshold:
                results.append({
                    "feature": query,
                    "score": round(confidence_score, 3)
                })

        # If no feature is detected above the threshold, return the most confident one (as a best guess)
        if not results and best_guess_feature:
            results.append({
                "feature": best_guess_feature['feature'],
                "score": best_guess_feature['score'],
                "note": "Low confidence, this is a best guess"
            })

        # Handle the case where no features or even best guesses are available
        if not results:
            return [{"feature": "Could not identify any accessibility features", "score": 0}]

        return results

    except FileNotFoundError:
        print(f"Error: The file {image_file} was not found.")
        return [{"feature": "error", "score": 0}]

    except Exception as e:
        print(f"Error during image classification: {e}")
        return [{"feature": "error", "score": 0}]
