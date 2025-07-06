"""
Groomify AI Chatbot Intents Configuration
Contains all intent definitions and keyword mappings for natural language processing
"""

# Basic intents with their associated keywords
BEAUTY_INTENTS = {
    "greeting": ["hello", "hi", "hey", "greetings"],
    "hairstyle": ["hairstyle", "haircut", "hair", "hairstyles", "haircuts"],
    "skincare": ["skincare routine", "skin care routine", "skincare advice", "skin care advice", "skincare", "skin care", "routine for skin"],
    "makeup": ["makeup", "cosmetics", "lipstick"],
    "hair_recommendation": ["hair recommendation", "hair suggest", "recommend hair", "suggest hairstyle", "hair advice", "what hairstyle", "hairstyle recommendation", "hairstyle suggest", "suggest me hair", "suggest hair", "hair style recommendation", "recommend hairstyle", "suggest me hairstyle"],
    "product_recommendation": ["product recommendation", "cosmetic recommendation", "recommend products", "suggest products", "product advice", "cosmetic advice", "product suggest", "cosmetics recommendation", "suggest me product", "makeup advice"],
    # Personal information queries - these should be checked first
    "ask_gender": ["what is my gender", "what gender am i", "my gender", "tell me my gender", "what gender", "am i male or female", "gender"],
    "ask_face_shape": ["what is my face shape", "what face shape am i", "my face shape", "tell me my face shape", "what face shape", "face shape"],
    "ask_skin_tone": ["what is my skin tone", "what skin tone am i", "my skin tone", "tell me my skin tone", "what skin tone", "skin tone"],
    "ask_skin_type": ["what is my skin type", "what skin type am i", "my skin type", "tell me my skin type", "what skin type", "skin type"],
    "ask_hair_type": ["what is my hair type", "what hair type am i", "my hair type", "tell me my hair type", "what hair type", "hair type"]
}

# Skin type keywords for NLP
SKIN_TYPES = {
    "dry": ["dry", "flaky", "tight", "rough", "dehydrated", "patchy", "scaly", "parched"],
    "normal": ["normal", "balanced", "healthy", "regular", "average", "combination"],
    "oily": ["oily", "greasy", "shiny", "slick", "acne-prone", "excessive sebum", "t-zone oil", "excess oil"]
}

# Face shape keywords for NLP
FACE_SHAPES = {
    "round": ["round", "circular", "full", "moon"],
    "oval": ["oval", "egg", "oblong", "long"],
    "square": ["square", "angular", "boxy", "rectangular"],
    "heart": ["heart", "pointed chin", "wide forehead", "heart-shaped"]
}

# Hair type keywords for NLP
HAIR_TYPES = {
    "straight": ["straight", "sleek", "flat"],
    "wavy": ["wavy", "waves", "beach waves"],
    "curly": ["curly", "curls", "coily"],
    "kinky": ["kinky", "coarse", "tight curls", "afro"],
    "dreadlocks": ["dreadlocks", "locs", "dreads"]
}
