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
            "dry": ["dry", "flaky", "tight", "rough", "dehydrated", "patchy", "scaly", "parched"],
            "normal": ["normal", "balanced", "healthy", "regular", "average", "combination"],
            "oily": ["oily", "greasy", "shiny", "slick", "acne-prone", "excessive sebum", "t-zone oil", "excess oil"]
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
        
        # Direct check for specific skin types explicitly mentioned
        if "oily skin" in text or "oily face" in text or "oily" in text or "greasy" in text:
            return "oily"
        if "dry skin" in text or "dry face" in text or "dry" in text or "flaky" in text:
            return "dry"
        if "normal skin" in text or "normal face" in text or "normal" in text:
            return "normal"
            
        # Secondary check using keyword lists
        for skin_type, keywords in self.SKIN_TYPES.items():
            if any(keyword in text for keyword in keywords):
                return skin_type
        
        # Check for partial matches and contextual cues
        if any(w in text for w in ["shine", "shiny", "acne", "pimples", "breakouts", "t-zone"]):
            return "oily"
        if any(w in text for w in ["flaking", "tight", "rough", "parched"]):
            return "dry"
        
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

    def format_analysis_result_html(self, analysis_data):
        """Format image analysis results as HTML for better presentation"""
        
        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 100%; margin: 0 auto; padding: 8px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); overflow: hidden;">
            <div style="text-align: center; margin-bottom: 12px;">
                <h2 style="color: #00665C; font-size: 16px; font-weight: bold; margin: 0; padding: 10px; background: rgba(255, 255, 255, 0.9); border-radius: 8px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); text-shadow: 0px 0px 1px rgba(0,0,0,0.2);">🎯 AI Beauty Analysis Results</h2>
            </div>
            
            <!-- Main characteristics display -->
            <div style="display: flex; flex-direction: column; gap: 8px; margin-bottom: 12px;">
                <!-- Gender & Face Shape -->
                <div style="display: flex; gap: 8px; width: 100%;">
                    <div style="flex: 1; background: white; border-radius: 10px; padding: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
                        <div style="font-size: 20px; margin-bottom: 4px;">👤</div>
                        <h3 style="color: #00665C; font-size: 11px; font-weight: 600; margin: 0 0 4px 0; text-transform: uppercase; letter-spacing: 0.5px;">Gender</h3>
                        <p style="color: #333; font-size: 13px; font-weight: bold; margin: 2px 0; text-transform: capitalize;">{analysis_data.get('gender', 'Unknown')}</p>
                        <p style="color: #666; font-size: 9px; margin: 2px 0 0 0; font-weight: 500;">{analysis_data.get('gender_confidence', 'N/A')}</p>
                    </div>
                    
                    <div style="flex: 1; background: white; border-radius: 10px; padding: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
                        <div style="font-size: 20px; margin-bottom: 4px;">🔍</div>
                        <h3 style="color: #00665C; font-size: 11px; font-weight: 600; margin: 0 0 4px 0; text-transform: uppercase; letter-spacing: 0.5px;">Face Shape</h3>
                        <p style="color: #333; font-size: 13px; font-weight: bold; margin: 2px 0; text-transform: capitalize;">{analysis_data.get('face_shape', 'Unknown')}</p>
                        <p style="color: #666; font-size: 9px; margin: 2px 0 0 0; font-weight: 500;">{analysis_data.get('face_confidence', 'N/A')}</p>
                    </div>
                </div>
                
                <!-- Hair Style & Skin Type -->
                <div style="display: flex; gap: 8px; width: 100%;">
                    <div style="flex: 1; background: white; border-radius: 10px; padding: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
                        <div style="font-size: 20px; margin-bottom: 4px;">💇</div>
                        <h3 style="color: #00665C; font-size: 11px; font-weight: 600; margin: 0 0 4px 0; text-transform: uppercase; letter-spacing: 0.5px;">Hair Style</h3>
                        <p style="color: #333; font-size: 13px; font-weight: bold; margin: 2px 0; text-transform: capitalize;">{analysis_data.get('hair_style', 'Unknown')}</p>
                        <p style="color: #666; font-size: 9px; margin: 2px 0 0 0; font-weight: 500;">{analysis_data.get('hair_confidence', 'N/A')}</p>
                    </div>
                    
                    <div style="flex: 1; background: white; border-radius: 10px; padding: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
                        <div style="font-size: 20px; margin-bottom: 4px;">🧴</div>
                        <h3 style="color: #00665C; font-size: 11px; font-weight: 600; margin: 0 0 4px 0; text-transform: uppercase; letter-spacing: 0.5px;">Skin Type</h3>
                        <p style="color: #333; font-size: 13px; font-weight: bold; margin: 2px 0; text-transform: capitalize;">{analysis_data.get('skin_type', 'Unknown')}</p>
                        <p style="color: #666; font-size: 9px; margin: 2px 0 0 0; font-weight: 500;">{analysis_data.get('skin_confidence', 'N/A')}</p>
                    </div>
                </div>
                
                <!-- Skin Tone -->
                <div style="background: white; border-radius: 10px; padding: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
                    <div style="font-size: 20px; margin-bottom: 4px;">🎨</div>
                    <h3 style="color: #00665C; font-size: 11px; font-weight: 600; margin: 0 0 4px 0; text-transform: uppercase; letter-spacing: 0.5px;">Skin Tone</h3>
                    <p style="color: #333; font-size: 13px; font-weight: bold; margin: 2px 0; text-transform: capitalize;">{analysis_data.get('skin_tone', 'Unknown')}</p>
                    <p style="color: #666; font-size: 9px; margin: 2px 0 0 0; font-weight: 500;">{analysis_data.get('skin_tone_confidence', 'N/A')}</p>
                </div>
            </div>
        """
        
        # Add hairstyle recommendations
        hairstyle_recs = analysis_data.get('hairstyle_recommendations', [])
        if hairstyle_recs:
            html += """
            <div style="background: white; border-radius: 10px; padding: 12px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
                <h3 style="color: #00665C; font-size: 14px; font-weight: bold; margin: 0 0 10px 0; padding-bottom: 6px; border-bottom: 2px solid #f0f0f0; display: flex; align-items: center;"><span style="margin-right: 6px;">💡</span> Recommended Hairstyles</h3>
                <div style="display: flex; flex-direction: column; gap: 6px;">
            """
            
            for idx, style in enumerate(hairstyle_recs, 1):
                # Skip any empty or 'nan' values
                if style == 'nan' or style == '' or not style:
                    continue
                    
                if isinstance(style, dict):
                    style_name = style.get('style', '')
                    style_desc = style.get('description', '')
                    
                    # Skip if style name is empty or nan
                    if not style_name or style_name == 'nan' or style_name == '':
                        continue
                    
                    html += f"""
                        <div style="padding: 8px; background: #f8f9fa; border-radius: 6px; border-left: 3px solid #00665C;">
                            <div style="display: flex; align-items: center; margin-bottom: 4px;">
                                <span style="background: #00665C; color: white; font-weight: bold; font-size: 10px; width: 18px; height: 18px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 8px; flex-shrink: 0;">{idx}</span>
                                <span style="color: #333; font-size: 13px; font-weight: 600;">{style_name}</span>
                            </div>
                    """
                    
                    # Only add description if it's valid (not empty or nan)
                    if style_desc and style_desc != 'nan' and style_desc != '':
                        html += f"""
                            <p style="color: #666; font-size: 11px; margin: 0 0 0 26px; line-height: 1.3;">{style_desc}</p>
                        """
                    
                    html += """
                        </div>
                    """
                else:
                    # For string style names, filter out 'nan' or empty values
                    if isinstance(style, str):
                        # Split to handle formats like "Style: Description"
                        style_parts = style.split(':', 1)
                        style_name = style_parts[0].strip()
                        
                        # Skip if style name is empty or nan
                        if not style_name or style_name == 'nan' or style_name == '':
                            continue
                            
                        html += f"""
                            <div style="display: flex; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 6px; border-left: 3px solid #00665C;">
                                <span style="background: #00665C; color: white; font-weight: bold; font-size: 10px; width: 18px; height: 18px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 8px; flex-shrink: 0;">{idx}</span>
                                <span style="color: #333; font-size: 13px; font-weight: 500;">{style_name}</span>
                        """
                        
                        # If description exists in the string, add it
                        if len(style_parts) > 1 and style_parts[1].strip() and style_parts[1].strip() != 'nan':
                            html += f"""
                                <p style="color: #666; font-size: 11px; margin: 8px 0 0 26px; line-height: 1.3;">{style_parts[1].strip()}</p>
                            """
                            
                        html += """
                            </div>
                        """
            html += "</div></div>"
        
        # Add product recommendations
        product_recs = analysis_data.get('product_recommendations', [])
        if product_recs:
            html += """
            <div style="background: white; border-radius: 10px; padding: 12px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
                <h3 style="color: #00665C; font-size: 14px; font-weight: bold; margin: 0 0 10px 0; padding-bottom: 6px; border-bottom: 2px solid #f0f0f0; display: flex; align-items: center;"><span style="margin-right: 6px;">🛍️</span> Recommended Products</h3>
                <div style="display: flex; flex-direction: column; gap: 6px;">
            """
            
            for idx, product in enumerate(product_recs[:5], 1):
                if isinstance(product, dict):
                    product_name = product.get('name', 'Unknown Product')
                    product_brand = product.get('brand', '')
                    product_price = product.get('price', '')
                    product_rating = product.get('rating', '')
                    
                    html += f"""
                        <div style="padding: 8px; background: #f8f9fa; border-radius: 6px; border-left: 3px solid #00665C;">
                    """
                    
                    if product_brand:
                        html += f"""
                            <div style="color: #00665C; font-size: 10px; font-weight: 500; margin-bottom: 2px;">{product_brand}</div>
                        """
                    
                    html += f"""
                            <div style="display: flex; align-items: center; justify-content: space-between;">
                                <div style="display: flex; align-items: center;">
                                    <span style="background: #00665C; color: white; font-weight: bold; font-size: 10px; width: 18px; height: 18px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 8px; flex-shrink: 0;">{idx}</span>
                                    <span style="color: #333; font-size: 13px; font-weight: 500;">{product_name}</span>
                                </div>
                    """
                    
                    if product_price or product_rating:
                        html += f"""
                                <div style="display: flex; align-items: center; gap: 8px;">
                        """
                        
                        if product_price:
                            html += f"""
                                    <span style="color: #27ae60; font-size: 11px; font-weight: 600;">{product_price}</span>
                            """
                        
                        if product_rating:
                            html += f"""
                                    <span style="color: #f39c12; font-size: 11px; font-weight: 600;">★ {product_rating}</span>
                            """
                        
                        html += """
                                </div>
                        """
                    
                    html += """
                            </div>
                        </div>
                    """
                else:
                    html += f"""
                        <div style="display: flex; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 6px; border-left: 3px solid #00665C;">
                            <span style="background: #00665C; color: white; font-weight: bold; font-size: 10px; width: 18px; height: 18px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 8px; flex-shrink: 0;">{idx}</span>
                            <span style="color: #333; font-size: 13px; font-weight: 500;">{product}</span>
                        </div>
                    """
            html += "</div></div>"
        
        html += """
            <div style="text-align: center; background: white; border-radius: 10px; padding: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
                <p style="color: #00665C; font-size: 13px; font-weight: 600; margin: 0;">✨ Ask me for more specific advice!</p>
            </div>
        </div>
        """
        
        return html

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
                        response = f"Based on your {skin_type} skin type, here are my recommendations:\n\n{routine}\n\n{tips}"
                        user_state.context = None
                    else:
                        return {
                            'intent': intent,
                            'response': "I didn't catch your skin type. Please specify if your skin is dry, normal, or oily? For example, 'I have oily skin' or 'My skin is dry'."
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
                    # Try to parse skin type from the current message using our enhanced parser
                    parsed_skin_info = self.parse_skincare_question(message_text)
                    
                    # If we extracted skin type from the message
                    if 'skin_type' in parsed_skin_info:
                        skin_type = parsed_skin_info['skin_type']
                        user_state.collected_info['skin_type'] = skin_type  # Save for future use
                        
                        # Prepare custom response if we found specific concerns
                        concern_intro = ""
                        if 'concerns' in parsed_skin_info and parsed_skin_info['concerns']:
                            concerns_list = ", ".join(parsed_skin_info['concerns'])
                            concern_intro = f"I see you're concerned about {concerns_list}. "
                        
                        routine = self.SKINCARE_ADVICE[skin_type]['routine']
                        tips = self.SKINCARE_ADVICE[skin_type]['tips']
                        response = f"{concern_intro}Based on your {skin_type} skin type, here are my recommendations:\n\n{routine}\n\n{tips}"
                    
                    # If no skin type in message but we know it from before
                    elif 'skin_type' in user_state.collected_info:
                        skin_type = user_state.collected_info['skin_type']
                        routine = self.SKINCARE_ADVICE[skin_type]['routine']
                        tips = self.SKINCARE_ADVICE[skin_type]['tips']
                        response = f"Based on your {skin_type} skin type, here are my recommendations:\n\n{routine}\n\n{tips}"
                    
                    # If we don't have the skin type, ask for it
                    else:
                        user_state.context = "collecting_skin_info"
                        response = "I'll help you with skincare advice! Please tell me your skin type - is it dry, normal, or oily? For example, 'I have oily skin' or 'My skin is dry'."
                        user_state.last_question = 'skin_type'
                        
                elif intent == "hairstyle":
                    # Check if we can parse a one-line hairstyle question first
                    parsed_info = self.parse_one_line_hairstyle_question(message_text)
                    print(f"Parsed hairstyle info: {parsed_info}")
                    
                    # If we have all the information needed, generate recommendations directly
                    if 'gender' in parsed_info and 'face_shape' in parsed_info and 'hair_type' in parsed_info:
                        # Use the extracted information to get recommendations
                        recommendations = self.hairstyle_recommender.recommend_hairstyle(
                            gender=parsed_info['gender'],
                            face_shape=parsed_info['face_shape'],
                            hair_type=parsed_info['hair_type']
                        )
                        
                        # Filter out empty or 'nan' recommendations
                        valid_recommendations = []
                        for rec in recommendations:
                            if isinstance(rec, dict):
                                if rec.get('style') and rec.get('style') != 'nan':
                                    valid_recommendations.append(rec)
                            elif rec and rec != 'nan':
                                valid_recommendations.append(rec)
                                
                        # Format response with recommendations
                        response = f"Based on your question, for {parsed_info['gender'].lower()} with a {parsed_info['face_shape']} face and {parsed_info['hair_type']} hair, I recommend these hairstyles:\n\n"
                        
                        if not valid_recommendations:
                            response += "I couldn't find specific recommendations for that combination. Would you like to try a different face shape or hair type?"
                        else:
                            for idx, rec in enumerate(valid_recommendations, 1):
                                if isinstance(rec, dict):
                                    style = rec.get('style', '')
                                    description = rec.get('description', '')
                                    
                                    if description and description != 'nan':
                                        response += f"{idx}. {style}: {description}\n\n"
                                    else:
                                        response += f"{idx}. {style}\n\n"
                                else:
                                    # Handle string format which might be "Style: Description"
                                    parts = rec.split(':', 1)
                                    response += f"{idx}. {parts[0].strip()}"
                                    if len(parts) > 1:
                                        response += f": {parts[1].strip()}\n\n"
                                    else:
                                        response += "\n\n"
                        
                        # Store the information for future use
                        user_state.collected_info.update(parsed_info)
                        
                        # No need to set context since we've already generated recommendations
                    else:
                        # Start collecting hairstyle information
                        user_state.context = "collecting_hair_info"
                        
                        # Pre-fill any information we've already extracted
                        if parsed_info:
                            user_state.collected_info.update(parsed_info)
                        
                        # If we already know gender from the question, ask for face shape next
                        if 'gender' in parsed_info and 'face_shape' in parsed_info:
                            user_state.last_question = 'hair_type'
                            response = f"What's your hair type? (straight, wavy, curly, kinky, or dreadlocks)"
                        elif 'gender' in parsed_info:
                            user_state.last_question = 'face_shape'
                            response = f"What's your face shape? (round, oval, square, or heart-shaped)"
                        else:
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
    
    def parse_one_line_hairstyle_question(self, message_text):
        """Parse a one-line hairstyle question to extract gender, face shape, and hair type"""
        info = {}
        message_text = message_text.lower()
        
        # Process with spaCy for better NLP understanding
        doc = self.nlp(message_text)
        
        # Check for gender mentions with word boundaries for more precise matching
        gender_keywords = {
            'Men': [r'\bman\b', r'\bmen\b', r'\bmale\b', r'\bguy\b', r'\bboy\b', r'\bhim\b', r'\bhis\b', r'\bgentleman\b'],
            'Women': [r'\bwoman\b', r'\bwomen\b', r'\bfemale\b', r'\bgirl\b', r'\blady\b', r'\bladies\b', r'\bher\b', r'\bshe\b']
        }
        
        import re
        for gender, patterns in gender_keywords.items():
            if any(re.search(pattern, message_text) for pattern in patterns):
                info['gender'] = gender
                break
                
        # If no gender explicitly mentioned but question suggests a gender from possessive 
        if 'gender' not in info:
            if "my wife" in message_text or "my girlfriend" in message_text or "my daughter" in message_text:
                info['gender'] = 'Women'
            elif "my husband" in message_text or "my boyfriend" in message_text or "my son" in message_text:
                info['gender'] = 'Men'
                
        # More comprehensive face shape detection with contextual understanding
        face_shape_patterns = {
            'round': [r'\bround face\b', r'\bround shaped\b', r'\bcircular face\b', r'\bfull face\b', r'\bmoon face\b', r'\bround\b'],
            'oval': [r'\boval face\b', r'\boval shaped\b', r'\boblong face\b', r'\begg shaped\b', r'\blong face\b', r'\boval\b'],
            'square': [r'\bsquare face\b', r'\bsquare shaped\b', r'\bangular face\b', r'\bstrong jaw\b', r'\brectangular\b', r'\bsquare\b'],
            'heart': [r'\bheart face\b', r'\bheart shaped\b', r'\bpointed chin\b', r'\bwide forehead\b', r'\bheart\b']
        }
        
        for face_shape, patterns in face_shape_patterns.items():
            if any(re.search(pattern, message_text) for pattern in patterns):
                info['face_shape'] = face_shape
                break
                
        # More comprehensive hair type detection with word boundaries
        hair_type_patterns = {
            'straight': [r'\bstraight hair\b', r'\bstraight\b', r'\bsleek hair\b', r'\bflat hair\b'],
            'wavy': [r'\bwavy hair\b', r'\bwaves\b', r'\bbeach waves\b', r'\bwavy\b'],
            'curly': [r'\bcurly hair\b', r'\bcurls\b', r'\bspiral\b', r'\bcurly\b'],
            'kinky': [r'\bkinky hair\b', r'\bcoarse hair\b', r'\btight curls\b', r'\bafro\b', r'\bkinky\b'],
            'dreadlocks': [r'\bdreadlocks\b', r'\blocs\b', r'\bdreads\b']
        }
        
        for hair_type, patterns in hair_type_patterns.items():
            if any(re.search(pattern, message_text) for pattern in patterns):
                info['hair_type'] = hair_type
                break
        
        # Look for entities that might have been missed
        for entity in doc.ents:
            # If we found a PERSON entity and gender is not determined, try to infer
            if entity.label_ == 'PERSON' and 'gender' not in info:
                # Use spaCy's entity recognition to try to determine gender from names
                # This is a simplification and not always accurate
                pass
        
        print(f"Parsed information from hairstyle question: {info}")
        return info
        
    def parse_skincare_question(self, message_text):
        """Parse a skincare question to extract skin type and concerns"""
        info = {}
        message_text = message_text.lower()
        
        # Process with spaCy
        doc = self.nlp(message_text)
        
        # Check for direct skin type mentions using regex for better accuracy
        import re
        skin_type_patterns = {
            'dry': [r'\bdry skin\b', r'\bdry type\b', r'\bskin is dry\b', r'\bhave dry\b', r'\bvery dry\b'],
            'oily': [r'\boily skin\b', r'\boily type\b', r'\bskin is oily\b', r'\bhave oily\b', r'\bvery oily\b', r'\bexcess oil\b'],
            'normal': [r'\bnormal skin\b', r'\bnormal type\b', r'\bskin is normal\b', r'\bhave normal\b']
        }
        
        # Check for direct mentions first
        for skin_type, patterns in skin_type_patterns.items():
            if any(re.search(pattern, message_text) for pattern in patterns):
                info['skin_type'] = skin_type
                break
                
        # If not found with direct patterns, use the keyword lists
        if 'skin_type' not in info:
            for skin_type, keywords in self.SKIN_TYPES.items():
                if any(keyword in message_text for keyword in keywords):
                    info['skin_type'] = skin_type
                    break
        
        # Check for skin concerns
        skin_concerns = []
        concern_keywords = {
            'acne': ['acne', 'pimple', 'breakout', 'spot', 'zit'],
            'aging': ['wrinkle', 'fine line', 'aging', 'mature', 'sagging'],
            'dark spots': ['dark spot', 'hyperpigmentation', 'discoloration', 'melasma'],
            'sensitivity': ['sensitive', 'irritation', 'redness', 'reaction', 'allergy']
        }
        
        for concern, keywords in concern_keywords.items():
            if any(keyword in message_text for keyword in keywords):
                skin_concerns.append(concern)
                
        if skin_concerns:
            info['concerns'] = skin_concerns
            
        print(f"Parsed information from skincare question: {info}")
        return info
