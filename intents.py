"""
Groomify AI Chatbot Intents Configuration
Contains all intent definitions and keyword mappings for natural language processing
"""

# Basic intents with their associated keywords (English + Roman Urdu)
BEAUTY_INTENTS = {
    "greeting": ["hello", "hi", "hey", "greetings", "salam", "assalam o alaikum", 
                 "assalamualaikum", "kya hal", "kya haal", "kaise hain", "kaise ho"],
    "hairstyle": ["hairstyle", "haircut", "hair", "hairstyles", "haircuts", 
                  "baal", "baalon", "hair style", "baal ka style", "baalon ka style", 
                  "hair cut", "baal katna", "hair design"],
    "skincare": ["skincare routine", "skin care routine", "skincare advice", 
                 "skin care advice", "skincare", "skin care", "routine for skin", 
                 "jild ki dekhbhal", "skin ki care", 
                 "face care", "chehre ki dekhbhal", "skin routine"],
    "hair_recommendation": ["hair recommendation", "hair suggest", "recommend hair", 
                            "suggest hairstyle", "hair advice", "what hairstyle", 
                            "hairstyle recommendation", "hairstyle suggest", 
                            "suggest me hair", "suggest hair", "hair style recommendation", 
                            "recommend hairstyle", "suggest me hairstyle", "baalon ka mashwara",
                            "hair ki salah", "baal suggest karo", "hair style batao",
                            "konsa hair style acha", "baal kaise rakhe"],
    "product_recommendation": ["product recommendation", "cosmetic recommendation", 
                               "recommend products", "suggest products", "product advice", 
                               "cosmetic advice", "product suggest", "cosmetics recommendation", 
                               "suggest me product", "makeup advice", "kya istemal karu", 
                               "konsa product acha", "makeup batao", "cosmetics suggest", 
                               "beauty product recommend"],
    # Personal information queries - these should be checked first (English + Roman Urdu)
    "ask_gender": ["what is my gender", "what gender am i", "my gender", "tell me my gender", "what gender", "am i male or female", "gender", "mera gender kya", "main mard ya aurat", "male hu ya female", "larki hu ya larka", "mera gender kya hai", "main kya hu"],
    "ask_face_shape": ["what is my face shape", "what face shape am i", "my face shape", "tell me my face shape", "what face shape", "face shape", "mera chehra kaisa", "face ka shape", "chehre ki shakl", "mera face shape kya", "chehra round ya oval", "mera chehra kaisa hai", "face shape kya hai"],
    "ask_skin_tone": ["what is my skin tone", "what skin tone am i", "my skin tone", "tell me my skin tone", "what skin tone", "skin tone", "mera rang kaisa", "skin ka color","mera skin tone", "complexion kaisa", "meri skin ka color", "mera rang kya hai", "skin ka color kya hai"],
    "ask_skin_type": ["what is my skin type", "what skin type am i", "my skin type", "tell me my skin type", "what skin type", "skin type", "meri skin kaisi","skin dry ya oily", "mera skin type kya", "jild ki qisam", "meri skin kaisi hai", "mera skin type", "skin kaisi hai"],
    "ask_hair_type": ["what is my hair type", "what hair type am i", "my hair type", "tell me my hair type", "what hair type", "hair type", "mere baal kaise", "hair ki type", "baal straight ya curly", "baalon ki qisam", "hair texture kaisa", "mere baal kaise hain", "baal ka type kya hai", "hair type kya hai"]
}

# Skin type keywords for NLP (English + Roman Urdu)
SKIN_TYPES = {
    "dry": ["dry", "flaky", "tight", "rough", "dehydrated", "patchy", "scaly", "sukhi", "sukha", "khushk", "sookhi skin", "dry skin"],
    "normal": ["normal", "balanced", "healthy", "regular", "average", "combination", "theek", "acha", "normal skin", "balanced skin", "sehat mand", "aam"],
    "oily": ["oily", "greasy", "shiny", "slick", "acne-prone", "excessive sebum", "t-zone oil", "excess oil", "chikna", "chamakdar", "tel wali", "oily skin", "tel zyada"]
}

# Face shape keywords for NLP (English + Roman Urdu)
FACE_SHAPES = {
    "round": ["round", "circular", "full", "moon", "gol", "gola", "circular face", "round face", "gol chehra", "chaand jaisa"],
    "oval": ["oval", "egg", "oblong", "long", "ande jaisa", "oval face", "long face", "lamba chehra"],
    "square": ["square", "angular", "boxy", "rectangular", "choras", "network", "square face", "kona dar", "chaukur"],
    "heart": ["heart", "pointed chin", "wide forehead", "heart-shaped", "dil jaisa", "heart shape"]
}

# Hair type keywords for NLP (English + Roman Urdu)
HAIR_TYPES = {
    "straight": ["straight", "sleek", "flat", "seedhe", "sidha", "straight hair", "seedhe baal", "sukhe baal"],
    "wavy": ["wavy", "waves", "beach waves", "lehradar", "laher", "wavy hair", "lehradar baal", "ghumdar"],
    "curly": ["curly", "curls", "coily", "ghungriale", "ghumdar", "curly hair", "ghungriale baal"],
    "kinky": ["kinky", "coarse", "tight curls", "afro", "sakht", "tangdu", "kinky hair", "sakht baal", "khaddar"],
    "dreadlocks": ["dreadlocks", "locs", "dreads", "jata", "choti", "dreadlocks hair", "lambe baal", "bandhe hue baal"]
}
