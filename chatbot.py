import spacy
from datetime import datetime
from hair_style_recommender import HairstyleRecommender

class UserState:
    def __init__(self):
        self.context = None  # Current context of conversation (e.g., "collecting_hair_info")
        self.collected_info = {}  # Store collected information
        self.last_question = None  # Last question asked by the bot
        self.last_analysis = None  # Timestamp of the last analysis performed

    def update_from_analysis(self, analysis_results):
        """Update user state with analysis results from image processing"""
        self.collected_info.update(analysis_results)
        self.last_analysis = datetime.utcnow().isoformat()
        # Reset context since we now have the information
        if self.context in ['collecting_skin_info', 'collecting_hair_info']:
            self.context = None

class GroomifyChat:
    def __init__(self):
        # Load English language model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # If model not found, download it
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.hairstyle_recommender = HairstyleRecommender()
        self.user_states = {}

        # Basic intents
        self.BEAUTY_INTENTS = {
            "greeting": ["hello", "hi", "hey", "greetings"],
            "hairstyle": ["hairstyle", "haircut", "hair", "hairstyles", "haircuts"],
            "skincare": ["skin", "skincare", "face", "complexion", "skin type", "skin care"],
            "makeup": ["makeup", "cosmetics", "lipstick"]
        }
        
        # Skin type keywords for NLP
        self.SKIN_TYPES = {
            "dry": ["dry", "flaky", "tight", "rough"],
            "normal": ["normal", "balanced", "healthy"],
            "oily": ["oily", "greasy", "shiny", "slick"]
        }

        # Face shape keywords for NLP
        self.FACE_SHAPES = {
            "round": ["round", "circular", "full", "moon"],
            "oval": ["oval", "egg", "oblong", "long"],
            "square": ["square", "angular", "boxy", "rectangular"],
            "heart": ["heart", "pointed chin", "wide forehead", "heart-shaped"]
        }

        # Hair type keywords for NLP
        self.HAIR_TYPES = {
            "straight": ["straight", "sleek", "flat"],
            "wavy": ["wavy", "waves", "beach waves"],
            "curly": ["curly", "curls", "coily"],
            "kinky": ["kinky", "coarse", "tight curls", "afro"],
            "dreadlocks": ["dreadlocks", "locs", "dreads"]
        }

        # Skincare advice templates
        self.SKINCARE_ADVICE = {
            "dry": {
                "routine": """<div class="analysis-message">
                    <h3>✨ Skincare Routine for Dry Skin</h3>
                    
                    <div class="analysis-section">
                        <h4>🌅 Morning Routine</h4>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 1</span>
                            <span class="detail-value">Gentle cream cleanser</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 2</span>
                            <span class="detail-value">Hydrating toner</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 3</span>
                            <span class="detail-value">Moisturizing serum (with hyaluronic acid)</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 4</span>
                            <span class="detail-value">Rich moisturizer</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 5</span>
                            <span class="detail-value">Sunscreen (SPF 30 or higher)</span>
                        </div>
                    </div>

                    <div class="analysis-section">
                        <h4>🌙 Evening Routine</h4>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 1</span>
                            <span class="detail-value">Oil-based cleanser</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 2</span>
                            <span class="detail-value">Hydrating toner</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 3</span>
                            <span class="detail-value">Nourishing serum</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 4</span>
                            <span class="detail-value">Night cream</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 5</span>
                            <span class="detail-value">Facial oil</span>
                        </div>
                    </div>

                    <div class="analysis-section">
                        <h4>📅 Weekly Treatments</h4>
                        <div class="analysis-detail">
                            <span class="detail-label">2-3x/week</span>
                            <span class="detail-value">Hydrating mask</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">1-2x/week</span>
                            <span class="detail-value">Gentle exfoliation</span>
                        </div>
                    </div>

                    <div class="analysis-section">
                        <h4>🧪 Key Ingredients to Look For</h4>
                        <div class="analysis-detail">
                            <span class="detail-value">• Hyaluronic Acid</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-value">• Ceramides</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-value">• Glycerin</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-value">• Squalane</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-value">• Natural oils (jojoba, argan, marula)</span>
                        </div>
                    </div>
                </div>""",
                
                "tips": """Tips for Dry Skin Care:

1. Never use hot water (it strips natural oils)
2. Apply moisturizer to damp skin
3. Use a humidifier while sleeping
4. Avoid alcohol-based products
5. Stay hydrated and eat omega-rich foods
6. Be gentle when towel-drying your face
7. Consider using overnight masks

Would you like specific product recommendations for any of these steps?"""
            },
            
            "normal": {
                "routine": """<div class="analysis-message">
                    <h3>✨ Skincare Routine for Normal Skin</h3>
                    
                    <div class="analysis-section">
                        <h4>🌅 Morning Routine</h4>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 1</span>
                            <span class="detail-value">Gentle water-based cleanser</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 2</span>
                            <span class="detail-value">Lightweight toner</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 3</span>
                            <span class="detail-value">Antioxidant serum (Vitamin C)</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 4</span>
                            <span class="detail-value">Light moisturizer</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 5</span>
                            <span class="detail-value">Sunscreen (SPF 30 or higher)</span>
                        </div>
                    </div>

                    <div class="analysis-section">
                        <h4>🌙 Evening Routine</h4>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 1</span>
                            <span class="detail-value">Cleanser</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 2</span>
                            <span class="detail-value">Toner</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 3</span>
                            <span class="detail-value">Treatment serum</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 4</span>
                            <span class="detail-value">Night moisturizer</span>
                        </div>
                    </div>

                    <div class="analysis-section">
                        <h4>📅 Weekly Treatments</h4>
                        <div class="analysis-detail">
                            <span class="detail-label">2-3x/week</span>
                            <span class="detail-value">Exfoliation</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">1-2x/week</span>
                            <span class="detail-value">Face mask</span>
                        </div>
                    </div>

                    <div class="analysis-section">
                        <h4>🧪 Key Ingredients to Look For</h4>
                        <div class="analysis-detail">
                            <span class="detail-value">• Niacinamide</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-value">• Vitamin C</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-value">• Peptides</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-value">• Hyaluronic Acid</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-value">• Antioxidants</span>
                        </div>
                    </div>
                </div>""",
                
                "tips": """Tips for Normal Skin Care:

1. Maintain consistent routine
2. Don't over-wash your face
3. Always remove makeup before bed
4. Use non-comedogenic products
5. Stay protected from sun damage
6. Keep skin hydrated
7. Listen to your skin's needs

Would you like product recommendations to maintain your healthy skin?"""
            },
            
            "oily": {
                "routine": """<div class="analysis-message">
                    <h3>✨ Skincare Routine for Oily Skin</h3>
                    
                    <div class="analysis-section">
                        <h4>🌅 Morning Routine</h4>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 1</span>
                            <span class="detail-value">Gel-based cleanser</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 2</span>
                            <span class="detail-value">Oil-control toner</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 3</span>
                            <span class="detail-value">Lightweight serum</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 4</span>
                            <span class="detail-value">Oil-free moisturizer</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 5</span>
                            <span class="detail-value">Mattifying sunscreen</span>
                        </div>
                    </div>

                    <div class="analysis-section">
                        <h4>🌙 Evening Routine</h4>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 1</span>
                            <span class="detail-value">Double cleanse (oil cleanser + foam cleanser)</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 2</span>
                            <span class="detail-value">Toner with salicylic acid</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 3</span>
                            <span class="detail-value">Treatment serum</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">Step 4</span>
                            <span class="detail-value">Light gel moisturizer</span>
                        </div>
                    </div>

                    <div class="analysis-section">
                        <h4>📅 Weekly Treatments</h4>
                        <div class="analysis-detail">
                            <span class="detail-label">2-3x/week</span>
                            <span class="detail-value">Clay mask</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-label">2x/week</span>
                            <span class="detail-value">Chemical exfoliant</span>
                        </div>
                    </div>

                    <div class="analysis-section">
                        <h4>🧪 Key Ingredients to Look For</h4>
                        <div class="analysis-detail">
                            <span class="detail-value">• Salicylic Acid</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-value">• Niacinamide</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-value">• Tea Tree Oil</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-value">• Hyaluronic Acid</span>
                        </div>
                        <div class="analysis-detail">
                            <span class="detail-value">• Clay</span>
                        </div>
                    </div>
                </div>""",
                
                "tips": """Tips for Oily Skin Care:

1. Don't skip moisturizer
2. Use oil-free products
3. Try chemical exfoliants
4. Don't over-wash your face
5. Use clay masks regularly
6. Consider using blotting papers
7. Stay hydrated

Would you like product recommendations for oil control?"""
            }
        }

    def get_face_shape_from_text(self, text):
        """Extract face shape from user's message"""
        text = text.lower()
        for shape, keywords in self.FACE_SHAPES.items():
            if any(keyword in text for keyword in keywords):
                return shape
        return None

    def get_hair_type_from_text(self, text):
        """Extract hair type from user's message"""
        text = text.lower()
        for hair_type, keywords in self.HAIR_TYPES.items():
            if any(keyword in text for keyword in keywords):
                return hair_type
        return None

    def get_skin_type_from_text(self, text):
        """Extract skin type from user's message"""
        text = text.lower()
        for skin_type, keywords in self.SKIN_TYPES.items():
            if any(keyword in text for keyword in keywords):
                return skin_type
        return None

    def detect_intent(self, text):
        """Detect the intent from user's message"""
        text = text.lower()
        for intent, keywords in self.BEAUTY_INTENTS.items():
            if any(keyword in text for keyword in keywords):
                return intent
        return "unknown"

    def generate_hairstyle_response(self, user_state):
        """Generate response based on collected hair information"""
        info = user_state.collected_info
        
        if 'gender' not in info:
            user_state.last_question = 'gender'
            return "Are you looking for men's or women's hairstyles?"
            
        if 'face_shape' not in info:
            user_state.last_question = 'face_shape'
            return "What's your face shape? (round, oval, square, heart)"
            
        if 'hair_type' not in info:
            user_state.last_question = 'hair_type'
            return "What's your hair type? (straight, wavy, curly, kinky, or dreadlocks)"
            
        # All information collected, get recommendations
        try:
            recommendations = self.hairstyle_recommender.recommend_hairstyle(
                gender=info['gender'],
                face_shape=info['face_shape'],
                hair_type=info['hair_type']
            )
            
            # Format recommendations into a nice message
            response = f"Based on your {info['face_shape']} face shape and {info['hair_type']} hair, here are some recommendations:\n\n"
            for rec in recommendations:
                response += f"• {rec['style']}: {rec['description']}\n"
            
            # Reset state since we're done
            user_state.context = None
            user_state.collected_info = {}
            
            return response
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return "I'm sorry, I had trouble getting recommendations. Please try again."

    def generate_skincare_advice(self, skin_type):
        """Generate skincare advice based on skin type"""
        advice = self.SKINCARE_ADVICE.get(skin_type)
        if not advice:
            return "I'm not sure how to help with that skin type. Can you tell me if your skin is dry, normal, or oily?"
        
        response = f"Skincare advice for {skin_type} skin:\n\n"
        response += advice["routine"] + "\n\n"
        response += advice["tips"]
        return response

    def handle_message(self, message_text, user_id="anonymous"):
        """Process a user message and return appropriate response"""
        try:
            # Get or create user state
            if user_id not in self.user_states:
                self.user_states[user_id] = UserState()
            user_state = self.user_states[user_id]

            # Process with spaCy
            doc = self.nlp(message_text.lower())
            
            # Detect intent
            intent = self.detect_intent(message_text)
            
            # Handle based on context and intent
            if user_state.context == "collecting_skin_info":
                if user_state.last_question == 'skin_type':
                    skin_type = self.get_skin_type_from_text(message_text)
                    if skin_type:
                        user_state.collected_info['skin_type'] = skin_type
                        routine = self.SKINCARE_ADVICE[skin_type]['routine']
                        tips = self.SKINCARE_ADVICE[skin_type]['tips']
                        response = f"{routine}\n\n{tips}"
                        user_state.context = None
                    else:
                        return {
                            'intent': intent,
                            'response': "I didn't catch your skin type. Is it dry, normal, or oily? You can also upload a photo for analysis."
                        }

            elif user_state.context == "collecting_hair_info":
                # Continue hair recommendation flow
                if user_state.last_question == 'gender':
                    if 'men' in message_text.lower() or 'male' in message_text.lower():
                        user_state.collected_info['gender'] = 'Men'
                    elif 'women' in message_text.lower() or 'female' in message_text.lower():
                        user_state.collected_info['gender'] = 'Women'
                    else:
                        return {
                            'intent': intent,
                            'response': "I didn't catch that. Are you looking for men's or women's hairstyles?"
                        }
                    
                elif user_state.last_question == 'face_shape':
                    face_shape = self.get_face_shape_from_text(message_text)
                    if face_shape:
                        user_state.collected_info['face_shape'] = face_shape
                    else:
                        return {
                            'intent': intent,
                            'response': "I'm not sure about that face shape. Is it round, oval, square, or heart-shaped?"
                        }
                    
                elif user_state.last_question == 'hair_type':
                    hair_type = self.get_hair_type_from_text(message_text)
                    if hair_type:
                        user_state.collected_info['hair_type'] = hair_type
                    else:
                        return {
                            'intent': intent,
                            'response': "I didn't catch your hair type. Is it straight, wavy, curly, kinky, or dreadlocks?"
                        }
                
                response = self.generate_hairstyle_response(user_state)
                
            else:
                # Handle initial intents
                if intent == "greeting":
                    response = "Hello! How can I help you with your beauty questions today?"
                    
                elif intent == "skincare":
                    if 'skin_type' in user_state.collected_info:
                        # If we already know their skin type from previous analysis
                        skin_type = user_state.collected_info['skin_type']
                        routine = self.SKINCARE_ADVICE[skin_type]['routine']
                        tips = self.SKINCARE_ADVICE[skin_type]['tips']
                        response = f"Based on your {skin_type} skin type, here are my recommendations:\n\n{routine}\n\n{tips}"
                    else:
                        user_state.context = "collecting_skin_info"
                        response = "I'll help you with skincare advice! First, do you know your skin type? Is it dry, normal, or oily? You can also upload a photo and I'll analyze it for you."
                        user_state.last_question = 'skin_type'
                        
                elif intent == "hairstyle":
                    user_state.context = "collecting_hair_info"
                    response = "I'll help you find the perfect hairstyle! First, are you looking for men's or women's hairstyles?"
                    user_state.last_question = 'gender'
                    
                elif intent == "makeup":
                    response = "What kind of makeup advice are you looking for? I can help with color selection and application tips."
                    
                else:
                    response = "I'm not sure what you're asking about. I can help with hairstyles, skincare, or makeup advice. What would you like to know?"

            return {
                'intent': intent,
                'response': response,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            print(f"Error processing message: {str(e)}")
            return {
                'intent': 'error',
                'response': "I'm sorry, I encountered an error. Please try again.",
                'timestamp': datetime.utcnow().isoformat()
            }
