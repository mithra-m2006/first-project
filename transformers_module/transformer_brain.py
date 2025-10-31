from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    pipeline
)
import torch

# 🔹 Model names
DIALOGUE_MODEL = "facebook/blenderbot-400M-distill"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"  # PyTorch only

print(f"🔹 Loading models...\n Dialogue: {DIALOGUE_MODEL}\n Emotion: {EMOTION_MODEL}")

# 🔹 Load the dialogue model (Blenderbot)
tokenizer = BlenderbotTokenizer.from_pretrained(DIALOGUE_MODEL)
model = BlenderbotForConditionalGeneration.from_pretrained(DIALOGUE_MODEL)

# 🔹 Load emotion classifier (PyTorch backend)
emotion_classifier = pipeline(
    "text-classification",
    model=EMOTION_MODEL,
    framework="pt"   # ✅ Force PyTorch so TF doesn't crash
)

# 🔹 Empathy prompt
EMPATHY_PROMPT = (
    "You are a gentle, curious, inner-voice AI that guides users by asking reflective questions.\n"
    "You help them think clearly and find strength, not just comfort.\n"
    "Your replies should be short, kind, and introspective — like their inner self speaking.\n\n"
)

def generate_empathic_reply(user_message: str, context: str) -> str:
    """
    Generate an empathic, reflective response to a user message.
    """

    # 🧠 Step 1: Detect emotion
    emotions = emotion_classifier(user_message)
    detected_emotion = emotions[0]['label']

    # 🧩 Step 2: Build enhanced context
    enhanced_context = f"{context}\n(User seems to be feeling {detected_emotion.lower()}.)"

    # 🪞 Step 3: Build inner-self style prompt
    prompt = (
        EMPATHY_PROMPT
        + f"Recent thoughts: {enhanced_context}\n\n"
        + f"User: {user_message}\n"
        + "Inner Self:"
    )

    # 🗣️ Step 4: Generate reply with Blenderbot
    inputs = tokenizer([prompt], return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ✨ Step 5: Extract clean reply after "Inner Self:"
    if "Inner Self:" in text:
        reply = text.split("Inner Self:")[-1].strip()
    else:
        reply = text.strip()

    # 🧭 Step 6: Make sure it’s concise
    return reply[:400]

# 🔍 Quick test
if __name__ == "__main__":
    user_input = "I’m feeling lonely and far from home."
    conversation_context = "The user mentioned they’re studying away from family."
    response = generate_empathic_reply(user_input, conversation_context)
    print("Inner Self:", response)
