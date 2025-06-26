from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from face_shape_model import FaceShapeDetector
from hair_style_model import HairStyleDetector
from gender_detection_model import GenderDetector
from skin_type_model import SkinTypeDetector
from skin_tone_model import SkinToneDetector
from chatbot import GroomifyChat, UserState
from product_recommender import ProductRecommender

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize models and chatbot
face_detector = FaceShapeDetector('models/face_shape_model.pth')
hair_detector = HairStyleDetector('models/hair_style_model.pth')
gender_detector = GenderDetector('models/gender_detection_model.pth')
skin_detector = SkinTypeDetector('models/skin_type_model.pth')
skin_tone_detector = SkinToneDetector('models/skintone_detection_model.pt')
product_recommender = ProductRecommender('datasets/cosmetics.csv')
chatbot = GroomifyChat()

# Store message IDs
message_counter = 0

def get_next_message_id():
    global message_counter
    message_counter += 1
    return message_counter

@app.route('/chat/message', methods=['POST'])
def handle_message():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        user_id = data.get('user_id', 'anonymous')
        message_text = data['message']

        # Process message through chatbot
        response_data = chatbot.handle_message(message_text, user_id)
        
        # Add message ID
        response_data['message_id'] = get_next_message_id()

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Save uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Get predictions using our detector classes
        face_shape, face_confidence = face_detector.detect_face_shape(filename)
        hair_style, hair_confidence = hair_detector.detect_hair_style(filename)
        gender, gender_confidence = gender_detector.detect_gender(filename)
        skin_type, skin_confidence = skin_detector.detect_skin_type(filename)
        skin_tone, skin_tone_confidence = skin_tone_detector.detect_skin_tone(filename)
        
        # Get hairstyle recommendations
        hairstyle_recommendations = chatbot.hairstyle_recommender.recommend_hairstyle(
            gender=gender,
            face_shape=face_shape,
            hair_type=hair_style
        )
        
        # Get product recommendations based on skin type
        product_recommendations = product_recommender.recommend_products(skin_type)

        # Prepare full analysis data for HTML formatting
        full_analysis_data = {
            'face_shape': face_shape,
            'face_confidence': f"{face_confidence:.2%}",
            'hair_style': hair_style,
            'hair_confidence': f"{hair_confidence:.2%}",
            'gender': gender,
            'gender_confidence': f"{gender_confidence:.2%}",
            'skin_type': skin_type,
            'skin_confidence': f"{skin_confidence:.2%}",
            'skin_tone': skin_tone,
            'skin_tone_confidence': f"{skin_tone_confidence:.2%}",
            'hairstyle_recommendations': hairstyle_recommendations,
            'product_recommendations': product_recommendations
        }
        
        # Generate HTML formatted analysis
        html_analysis = chatbot.format_analysis_result_html(full_analysis_data)
        
        # Update chatbot state with analysis results
        user_id = request.args.get('user_id', 'anonymous')
        if user_id not in chatbot.user_states:
            chatbot.user_states[user_id] = UserState()
        
        user_state = chatbot.user_states[user_id]
        analysis_results = {
            'face_shape': face_shape,
            'hair_style': hair_style,
            'gender': gender,
            'skin_type': skin_type,
            'skin_tone': skin_tone
        }
        user_state.update_from_analysis(analysis_results)
        
        # Delete the image file after analysis for privacy and security
        try:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Successfully deleted image file: {filename}")
        except Exception as delete_error:
            print(f"Error deleting image file {filename}: {str(delete_error)}")
            
        return jsonify({
            'face_shape': face_shape,
            'face_confidence': f"{face_confidence:.2%}",
            'hair_style': hair_style,
            'hair_confidence': f"{hair_confidence:.2%}",
            'gender': gender,
            'gender_confidence': f"{gender_confidence:.2%}",
            'skin_type': skin_type,
            'skin_confidence': f"{skin_confidence:.2%}",
            'skin_type_confidence': f"{skin_confidence:.2%}",
            'skin_tone': skin_tone,
            'skin_tone_confidence': f"{skin_tone_confidence:.2%}",
            # Don't return image_path since we've deleted the file
            'hairstyle_recommendations': hairstyle_recommendations,
            'product_recommendations': product_recommendations,
            'html_analysis': html_analysis
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<path:filename>')
def serve_image(filename):
    # Check if file exists, since we delete files after analysis
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    else:
        return jsonify({'error': 'Image no longer available'}), 404

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat/reset', methods=['POST'])
def reset_chat():
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'anonymous')
        
        # Reset user state in chatbot
        if user_id in chatbot.user_states:
            chatbot.user_states[user_id] = UserState()
        
        return jsonify({
            'success': True,
            'message': "Hello! I'm Groomify AI Assistant. I can help you with hairstyles, skincare, and makeup advice. Click the Help button to see what I can do!"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Groomify AI Backend is running',
        'version': '1.0.0'
    }), 200

if __name__ == '__main__':
    print("🚀 Starting Groomify AI Backend...")
    print("📍 Server will be available at: http://localhost:5000")
    print("📍 For React Native testing, use your IP address instead of localhost")
    app.run(debug=True, host='0.0.0.0', port=5000)