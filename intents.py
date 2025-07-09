# """
# Groomify AI Chatbot Intents Configuration
# Contains all intent definitions and keyword mappings for natural language processing
# """

# # Basic intents with their associated keywords (English + Roman Urdu)
# BEAUTY_INTENTS = {
#     "greeting": ["hello", "hi", "hey", "greetings", "salam", "assalam o alaikum", 
#                  "assalamualaikum", "kya hal", "kya haal", "kaise hain", "kaise ho"],
#     "hairstyle": ["hairstyle", "haircut", "hair", "hairstyles", "haircuts", 
#                   "baal", "baalon", "hair style", "baal ka style", "baalon ka style", 
#                   "hair cut", "baal katna", "hair design"],
#     "skincare": ["skincare routine", "skin care routine", "skincare advice", 
#                  "skin care advice", "skincare", "skin care", "routine for skin", 
#                  "jild ki dekhbhal", "skin ki care", 
#                  "face care", "chehre ki dekhbhal", "skin routine"],
#     "hair_recommendation": ["hair recommendation", "hair suggest", "recommend hair", 
#                             "suggest hairstyle", "hair advice", "what hairstyle", 
#                             "hairstyle recommendation", "hairstyle suggest", 
#                             "suggest me hair", "suggest hair", "hair style recommendation", 
#                             "recommend hairstyle", "suggest me hairstyle", "baalon ka mashwara",
#                             "hair ki salah", "baal suggest karo", "hair style batao",
#                             "konsa hair style acha", "baal kaise rakhe"],
#     "product_recommendation": ["product recommendation", "cosmetic recommendation", 
#                                "recommend products", "suggest products", "product advice", 
#                                "cosmetic advice", "product suggest", "cosmetics recommendation", 
#                                "suggest me product", "makeup advice", "kya istemal karu", 
#                                "konsa product acha", "makeup batao", "cosmetics suggest", 
#                                "beauty product recommend"],
#     # Personal information queries - these should be checked first (English + Roman Urdu)
#     "ask_gender": ["what is my gender", "what gender am i", "my gender", "tell me my gender", "what gender", "am i male or female", "gender", "mera gender kya", "main mard ya aurat", "male hu ya female", "larki hu ya larka", "mera gender kya hai", "main kya hu"],
#     "ask_face_shape": ["what is my face shape", "what face shape am i", "my face shape", "tell me my face shape", "what face shape", "face shape", "mera chehra kaisa", "face ka shape", "chehre ki shakl", "mera face shape kya", "chehra round ya oval", "mera chehra kaisa hai", "face shape kya hai"],
#     "ask_skin_tone": ["what is my skin tone", "what skin tone am i", "my skin tone", "tell me my skin tone", "what skin tone", "skin tone", "mera rang kaisa", "skin ka color","mera skin tone", "complexion kaisa", "meri skin ka color", "mera rang kya hai", "skin ka color kya hai"],
#     "ask_skin_type": ["what is my skin type", "what skin type am i", "my skin type", "tell me my skin type", "what skin type", "skin type", "meri skin kaisi","skin dry ya oily", "mera skin type kya", "jild ki qisam", "meri skin kaisi hai", "mera skin type", "skin kaisi hai"],
#     "ask_hair_type": ["what is my hair type", "what hair type am i", "my hair type", "tell me my hair type", "what hair type", "hair type", "mere baal kaise", "hair ki type", "baal straight ya curly", "baalon ki qisam", "hair texture kaisa", "mere baal kaise hain", "baal ka type kya hai", "hair type kya hai"]
# }

# # Skin type keywords for NLP (English + Roman Urdu)
# SKIN_TYPES = {
#     "dry": ["dry", "flaky", "tight", "rough", "dehydrated", "patchy", "scaly", "sukhi", "sukha", "khushk", "sookhi skin", "dry skin"],
#     "normal": ["normal", "balanced", "healthy", "regular", "average", "combination", "theek", "acha", "normal skin", "balanced skin", "sehat mand", "aam"],
#     "oily": ["oily", "greasy", "shiny", "slick", "acne-prone", "excessive sebum", "t-zone oil", "excess oil", "chikna", "chamakdar", "tel wali", "oily skin", "tel zyada"]
# }

# # Face shape keywords for NLP (English + Roman Urdu)
# FACE_SHAPES = {
#     "round": ["round", "circular", "full", "moon", "gol", "gola", "circular face", "round face", "gol chehra", "chaand jaisa"],
#     "oval": ["oval", "egg", "oblong", "long", "ande jaisa", "oval face", "long face", "lamba chehra"],
#     "square": ["square", "angular", "boxy", "rectangular", "choras", "network", "square face", "kona dar", "chaukur"],
#     "heart": ["heart", "pointed chin", "wide forehead", "heart-shaped", "dil jaisa", "heart shape"]
# }

# # Hair type keywords for NLP (English + Roman Urdu)
# HAIR_TYPES = {
#     "straight": ["straight", "sleek", "flat", "seedhe", "sidha", "straight hair", "seedhe baal", "sukhe baal"],
#     "wavy": ["wavy", "waves", "beach waves", "lehradar", "laher", "wavy hair", "lehradar baal", "ghumdar"],
#     "curly": ["curly", "curls", "coily", "ghungriale", "ghumdar", "curly hair", "ghungriale baal"],
#     "kinky": ["kinky", "coarse", "tight curls", "afro", "sakht", "tangdu", "kinky hair", "sakht baal", "khaddar"],
#     "dreadlocks": ["dreadlocks", "locs", "dreads", "jata", "choti", "dreadlocks hair", "lambe baal", "bandhe hue baal"]
# }
"""
Groomify AI Chatbot Intents Configuration
Enhanced English-only version with extensive keyword coverage and spelling variations
"""

# Basic intents with their associated keywords (English with spelling variations)
BEAUTY_INTENTS = {
    "greeting": [
        "hello", "hi", "hey", "greetings", "hi there", "hello there", 
        "good morning", "good afternoon", "good evening", "howdy", 
        "what's up", "whats up", "sup", "yo", "hey there", "hiya",
        "greets", "how are you", "how do you do", "how's it going",
        "how goes it", "good day", "good to see you", "welcome",
        "hi friend", "hello dear", "hey buddy", "hey you", "hi bot",
        "hello chatbot", "hi assistant", "hey beauty bot"
    ],
    
    "hairstyle": [
        "hairstyle", "hair style", "haircut", "hair cut", "hairdo", 
        "hair do", "hair look", "hair fashion", "hair trend", 
        "hair design", "hair makeover", "new hairstyle", "latest hairstyle", 
        "trending hairstyle", "popular hairstyle", "short hairstyle", 
        "long hairstyle", "medium hairstyle", "formal hairstyle", 
        "casual hairstyle", "party hairstyle", "wedding hairstyle", 
        "professional hairstyle", "hair transformation", "hair appearance",
        "hair shaping", "hair dressing", "hair arrangement", "hair set",
        "hair lookbook", "hair inspiration", "hair ideas", "hair options",
        "hair possibilities", "hair suggestions", "hair catalog",
        "hair gallery", "hair portfolio", "hair reference", "hair examples"
    ],
    
    "skincare": [
        "skincare", "skin care", "skincare routine", "skin care routine", 
        "skincare advice", "skin care advice", "routine for skin", 
        "skin regimen", "skin maintenance", "skin health", "skin treatment", 
        "face care", "facial care", "complexion care", "daily skincare", 
        "nighttime skincare", "morning skincare", "skin protection", 
        "skin management", "skin preservation", "skin conditioning", 
        "skin nourishment", "skin hydration", "skin moisturization", 
        "skin cleansing", "skin toning", "skin exfoliation", "skin regime", 
        "skin program", "skin schedule", "skin process", "skin steps", 
        "skin methodology", "skin approach", "skin system", "skin practice"
    ],
    
    "hair_recommendation": [
        "hair recommendation", "hair suggest", "recommend hair", 
        "suggest hairstyle", "hair advice", "what hairstyle", 
        "hairstyle recommendation", "hairstyle suggest", "suggest me hair", 
        "suggest hair", "hair style recommendation", "recommend hairstyle", 
        "suggest me hairstyle", "best hairstyle for me", "perfect hairstyle", 
        "ideal hairstyle", "what haircut should I get", "which hairstyle suits me", 
        "hairstyle for my face", "hairstyle according to face shape", 
        "trending hairstyle for me", "hairstyle that fits me", 
        "hairstyle that suits me", "hairstyle that flatters me", 
        "hairstyle that looks good on me", "hairstyle that works for me", 
        "hairstyle that complements me", "hairstyle that matches me", 
        "hairstyle that enhances me", "hairstyle that fits my face", 
        "hairstyle that suits my face", "hairstyle for my features", 
        "hairstyle for my personality", "hairstyle for my age", 
        "hairstyle for my profession", "hairstyle for my lifestyle"
    ],
    
    "product_recommendation": [
        "product recommendation", "product suggest", "recommend products","product",
        "suggest products", "product advice", "cosmetic advice", 
        "product suggestions", "cosmetics recommendation", "suggest me product", 
        "makeup advice", "beauty products", "skincare products", 
        "haircare products", "best products for", "top products", 
        "product ratings", "product reviews", "what should I buy", 
        "which product is best", "product options", "product choices", 
        "product alternatives", "product possibilities", "product solutions", 
        "product selections", "product picks", "product favorites", 
        "product must-haves", "product essentials", "product necessities", 
        "product requirements", "product needs", "product desires", 
        "product wishes", "product preferences", "product criteria"
    ],
    
    # Personal information queries - extensive keywords (English with variations)
    "ask_gender": [
        "what is my gender", "what gender am i", "my gender", 
        "tell me my gender", "what gender", "am i male or female", 
        "gender", "my sex", "what sex am i", "am i a man or woman", 
        "am i boy or girl", "identify my gender", "determine my gender", 
        "check my gender", "verify my gender", "gender check", 
        "gender identification", "gender analysis", "gender assessment", 
        "gender evaluation", "gender determination", "gender recognition", 
        "gender classification", "gender categorization", "gender typing", 
        "gender specification", "gender clarification", "gender confirmation", 
        "gender verification", "gender validation", "gender authentication"
    ],
    
    "ask_face_shape": [
        "what is my face shape", "what face shape am i", "my face shape", 
        "tell me my face shape", "what face shape", "face shape", "face type", 
        "shape of my face", "determine face shape", "identify face shape", 
        "check face shape", "analyze face shape", "what's my face structure", 
        "how is my face shaped", "face shape analysis", "face shape recognition", 
        "face shape detection", "face shape identification", "face shape evaluation", 
        "face shape assessment", "face shape classification", "face shape categorization", 
        "face shape determination", "face shape verification", "face shape clarification", 
        "face shape specification", "face shape typing", "face shape confirmation", 
        "face shape validation", "face shape authentication"
    ],
    
    "ask_skin_tone": [
        "what is my skin tone", "what skin tone am i", "my skin tone", 
        "tell me my skin tone", "what skin tone", "skin tone", "complexion", 
        "skin color", "skin shade", "skin pigmentation", "determine skin tone", 
        "identify skin tone", "check skin tone", "analyze skin tone", 
        "what's my complexion", "how would you describe my skin color", 
        "skin tone finder", "skin tone analyzer", "skin tone chart", 
        "skin tone detection", "skin tone identification", "skin tone evaluation", 
        "skin tone assessment", "skin tone classification", "skin tone categorization", 
        "skin tone determination", "skin tone verification", "skin tone clarification", 
        "skin tone specification", "skin tone typing", "skin tone confirmation", 
        "skin tone validation", "skin tone authentication"
    ],
    
    "ask_skin_type": [
        "what is my skin type", "what skin type am i", "my skin type", 
        "tell me my skin type", "what skin type", "skin type", "skin condition", 
        "skin nature", "skin characteristics", "determine skin type", 
        "identify skin type", "check skin type", "analyze skin type", 
        "what's my skin like", "how would you describe my skin", 
        "skin type finder", "skin type analyzer", "skin type test", 
        "skin type detection", "skin type identification", "skin type evaluation", 
        "skin type assessment", "skin type classification", "skin type categorization", 
        "skin type determination", "skin type verification", "skin type clarification", 
        "skin type specification", "skin type typing", "skin type confirmation", 
        "skin type validation", "skin type authentication"
    ],
    
    "ask_hair_type": [
        "what is my hair type", "what hair type am i", "my hair type", 
        "tell me my hair type", "what hair type", "hair type", "hair texture", 
        "hair nature", "hair characteristics", "determine hair type", 
        "identify hair type", "check hair type", "analyze hair type", 
        "what's my hair like", "how would you describe my hair", 
        "hair type finder", "hair type analyzer", "hair type test", 
        "hair type detection", "hair type identification", "hair type evaluation", 
        "hair type assessment", "hair type classification", "hair type categorization", 
        "hair type determination", "hair type verification", "hair type clarification", 
        "hair type specification", "hair type typing", "hair type confirmation", 
        "hair type validation", "hair type authentication"
    ]
}

# Enhanced skin type keywords for NLP (English with variations)
SKIN_TYPES = {
    "dry": [
        "dry", "flaky", "flaking", "tight", "tightness", "rough", "roughness",
        "dehydrated", "dehydration", "patchy", "patches", "scaly", "scales",
        "peeling", "peels", "cracked", "cracks", "itchy", "itching", "irritated",
        "irritation", "lacking moisture", "moisture loss", "parched", "parchedness",
        "sensitive", "sensitivity", "easily irritated", "weather affected",
        "weather sensitive", "dryness", "dry skin", "skin dryness", "lacking hydration",
        "needs moisture", "requires hydration", "moisture deficient", "water loss",
        "transepidermal water loss", "desiccated", "arid", "thirsty skin"
    ],
    
    "normal": [
        "normal", "balanced", "balance", "healthy", "health", "regular", "regularity",
        "average", "combination", "not too oily", "not too dry", "well-balanced",
        "well balanced", "stable", "stability", "consistent", "consistency",
        "problem-free", "problem free", "clear", "clarity", "even", "evenness",
        "smooth", "smoothness", "soft", "softness", "comfortable", "comfort",
        "normal skin", "balanced skin", "healthy skin", "regular skin",
        "average skin", "combination skin", "stable skin", "consistent skin",
        "clear skin", "even skin", "smooth skin", "soft skin", "comfortable skin"
    ],
    
    "oily": [
        "oily", "oiliness", "greasy", "greasiness", "shiny", "shine", "slick",
        "slickness", "acne-prone", "acne prone", "excessive sebum", "excess sebum",
        "t-zone oil", "t zone oil", "excess oil", "shiny forehead", "shiny nose",
        "shiny chin", "greasy feeling", "oil slick", "shiny complexion",
        "breakout prone", "enlarged pores", "large pores", "blackheads", "black heads",
        "whiteheads", "white heads", "shiny t-zone", "oily skin", "skin oiliness",
        "excess shine", "greasy skin", "slick skin", "shiny skin", "oily complexion",
        "sebum overproduction", "overactive sebaceous glands", "oil production"
    ],
    
    "combination": [
        "combination", "combo", "mixed", "both oily and dry", "oily and dry",
        "t-zone oily", "t zone oily", "cheeks dry", "oily in some areas",
        "dry in some areas", "uneven skin", "unevenness", "oily t-zone",
        "dry cheeks", "mixed skin type", "combination type", "combination skin",
        "combo skin", "mixed skin", "oily dry combination", "oily and dry areas",
        "varying skin types", "different zones", "t-zone differences", "zone specific",
        "combination complexion", "mixed complexion", "dual skin type", "two types"
    ],
    
    "sensitive": [
        "sensitive", "sensitivity", "reactive", "reactivity", "easily irritated",
        "irritation prone", "redness", "red", "stinging", "sting", "burning",
        "burn", "allergic", "allergies", "fragrance-sensitive", "fragrance sensitive",
        "product reactions", "reacts to products", "easily red", "rosacea",
        "eczema", "dermatitis", "flushing", "flush", "delicate", "delicacy",
        "fragile", "fragility", "thin-skinned", "thin skinned", "sensitive skin",
        "reactive skin", "irritated skin", "red skin", "stinging skin", "burning skin",
        "allergic skin", "fragrance sensitive skin", "delicate skin", "fragile skin"
    ]
}

# Enhanced face shape keywords for NLP (English with variations)
FACE_SHAPES = {
    "round": [
        "round", "rounded", "circular", "circle", "full", "fullness", "moon",
        "moon shaped", "soft angles", "soft angled", "wide", "width", "equal length",
        "equal width", "full cheeks", "rounded jaw", "no sharp angles", "round face",
        "rounded face", "circular face", "full face", "moon face", "soft angled face",
        "wide face", "equal proportions", "full cheeked", "rounded jawline",
        "no angles", "curved features", "soft contours", "circular shape"
    ],
    
    "oval": [
        "oval", "ovular", "egg", "egg shaped", "oblong", "oblong shape", "long",
        "elongated", "balanced", "balance", "proportionate", "proportions",
        "tapered chin", "slightly wider forehead", "gentle curves", "most versatile",
        "classic shape", "slightly longer", "oval face", "ovular face", "egg face",
        "oblong face", "long face", "elongated face", "balanced face", "proportionate face",
        "tapered chin face", "wide forehead face", "gentle curved face", "versatile face",
        "classic face shape", "longer face", "balanced proportions", "symmetrical"
    ],
    
    "square": [
        "square", "squared", "angular", "angle", "boxy", "box like", "rectangular",
        "rectangle", "strong jaw", "wide forehead", "straight sides", "sharp angles",
        "defined jawline", "equal width", "forehead cheeks jaw same", "square face",
        "squared face", "angular face", "boxy face", "rectangular face", "strong jaw face",
        "wide forehead face", "straight sided face", "sharp angled face", "defined jawline face",
        "equal width face", "uniform width", "angular features", "sharp jawline",
        "pronounced angles", "geometric shape", "architectural features"
    ],
    
    "heart": [
        "heart", "heart shaped", "pointed chin", "wide forehead", "broad forehead",
        "narrow chin", "prominent cheekbones", "widest at temples", "tapering",
        "tapered", "inverted triangle", "widest at top", "heart face", "heart shaped face",
        "pointed chin face", "wide forehead face", "broad forehead face", "narrow chin face",
        "prominent cheekbones face", "temple wide face", "tapering face", "inverted triangle face",
        "widest top face", "chin narrows", "forehead dominant", "cheekbone emphasis",
        "facial taper", "triangle shape", "upside down triangle", "v-shaped"
    ],
    
    "diamond": [
        "diamond", "diamond shaped", "angular", "angle", "wide cheekbones", "narrow forehead",
        "narrow chin", "pointed chin", "high cheekbones", "elongated", "face angular",
        "widest at cheekbones", "geometric", "diamond face", "diamond shaped face",
        "angular face", "wide cheekbone face", "narrow forehead face", "narrow chin face",
        "pointed chin face", "high cheekbone face", "elongated face", "angular face shape",
        "widest cheekbone face", "geometric face", "pronounced cheekbones", "narrow at ends",
        "wide middle", "angular structure", "diamond structure", "faceted shape"
    ],
    
    "oblong": [
        "oblong", "oblong shaped", "long", "lengthy", "rectangular", "rectangle", "elongated",
        "extended", "straight sides", "rounded corners", "equal width", "forehead cheeks jaw same",
        "length greater than width", "stretched", "oblong face", "long face", "rectangular face",
        "elongated face", "extended face", "straight sided face", "rounded corner face",
        "equal width face", "uniform width face", "longer length", "stretched face",
        "vertical emphasis", "length dominant", "rectilinear", "linear shape", "extended shape"
    ]
}

# Enhanced hair type keywords for NLP (English with variations)
HAIR_TYPES = {
    "straight": [
        "straight", "straight", "stright", "sleek", "sleak", "flat", "flattened",
        "smooth", "smoothe", "silky", "silkey", "shiny", "shiney", "pin-straight",
        "pin straight", "straight as pin", "no curl", "no wave", "flat ironed",
        "flat-ironed", "naturally straight", "stick straight", "straight and fine",
        "straight and thick", "straight and limp", "straight hair", "sleek hair",
        "flat hair", "smooth hair", "silky hair", "shiny hair", "pin-straight hair",
        "straightened hair", "uncurled", "non-curly", "non-wavy", "lank", "lanky"
    ],
    
    "wavy": [
        "wavy", "wavey", "waves", "wave", "beach waves", "beach wave", "natural waves",
        "natural wave", "loose waves", "loose wave", "s-shaped", "s shaped", "undulating",
        "undulation", "gentle waves", "gentle wave", "soft waves", "soft wave", "body waves",
        "body wave", "textured waves", "textured wave", "wave pattern", "not straight",
        "not curly", "in-between", "in between", "slight curl", "wavy hair", "wavey hair",
        "beach wave hair", "natural wave hair", "loose wave hair", "s-shaped hair",
        "undulating hair", "textured hair", "natural texture", "soft curl", "loose curl"
    ],
    
    "curly": [
        "curly", "curley", "curl", "curls", "coily", "coil", "coils", "spiral", "spirals",
        "ringlets", "ringlet", "tight curls", "tight curl", "loose curls", "loose curl",
        "natural curls", "natural curl", "defined curls", "defined curl", "bouncy curls",
        "bouncy curl", "springy", "spring", "curl pattern", "s-shaped curls", "s shaped curls",
        "corkscrew", "corkscrew curl", "frizzy curls", "frizzy curl", "dry curls", "dry curl",
        "voluminous curls", "voluminous curl", "curly hair", "curl hair", "coily hair",
        "spiral hair", "ringlet hair", "tight curl hair", "natural curl hair", "defined curl hair"
    ],
    
    "kinky": [
        "kinky", "kink", "kinks", "coarse", "coars", "tight curls", "tight curl", "afro",
        "afro hair", "zigzag", "zig zag", "very tight curls", "very tight curl", "type 4 hair",
        "type four hair", "coily hair", "coil hair", "nappy", "nappie", "textured", "texture",
        "shrinks when dry", "shrink when dry", "fragile", "fragil", "dry texture", "dry textures",
        "voluminous", "volume", "dense", "dens", "springy coils", "springy coil", "kinky hair",
        "kink hair", "coarse hair", "tight curl hair", "afro textured", "zigzag pattern",
        "type 4 texture", "nappy hair", "shrinkage hair", "fragile hair", "dense hair"
    ],
    
    "dreadlocks": [
        "dreadlocks", "dread locks", "dreads", "dread", "locs", "loc", "matted hair", "mat hair",
        "ropes", "rope", "locked hair", "lock hair", "freeform locs", "free form locs", "cultivated locs",
        "cultivated locks", "sisterlocks", "sister locks", "mature locs", "mature locks", "new locs",
        "new locks", "budding locs", "budding locks", "interlocked", "interlock", "palm rolled",
        "palm roll", "twisted", "twist", "dreadlock hair", "dread hair", "locked hair", "matte hair",
        "rope hair", "freeform hair", "cultivated hair", "sisterlocked hair", "mature dreads",
        "interlocked hair", "palm rolled hair", "twisted hair", "knotted hair"
    ],
    
    "thin": [
        "thin", "thin hair", "fine", "fine hair", "limp", "limp hair", "flat", "flat hair",
        "sparse", "sparse hair", "not thick", "not thick hair", "delicate", "delicate hair",
        "see-through", "see through", "light", "light hair", "wispy", "wispy hair", "baby fine",
        "baby fine hair", "lack volume", "no volume", "flyaways", "flyaway", "hard to style",
        "difficult to style", "oily quickly", "gets oily fast", "thin strands", "fine strands",
        "limp strands", "flat strands", "sparse strands", "delicate strands", "light strands",
        "wispy strands", "baby fine strands", "low density", "low density hair", "fine texture"
    ],
    
    "thick": [
        "thick", "thick hair", "dense", "dense hair", "full", "full hair", "voluminous",
        "voluminous hair", "heavy", "heavy hair", "coarse", "coarse hair", "luxuriant",
        "luxuriant hair", "abundant", "abundant hair", "bulky", "bulky hair", "hard to manage",
        "difficult to manage", "lots of hair", "much hair", "full-bodied", "full bodied",
        "thick strands", "dense strands", "full strands", "voluminous strands", "heavy strands",
        "coarse strands", "luxuriant strands", "abundant strands", "bulky strands", "high density",
        "high density hair", "coarse texture", "thick texture", "dense growth", "full growth"
    ]
}