import pandas as pd
import os

class HairstyleRecommender:
    def __init__(self, csv_path='datasets/hairstyle_recommendations.csv'):
        """
        Initialize the hairstyle recommender with a CSV dataset.
        
        Args:
            csv_path (str): Path to the hairstyle recommendations CSV file.
        """
        self.csv_path = csv_path
        self.df = None
        self.load_data()

    def load_data(self):
        """Load the hairstyle recommendations data from CSV."""
        try:
            self.df = pd.read_csv(self.csv_path)
        except Exception as e:
            print(f"Error loading recommendations data: {e}")
            self.df = None

    def recommend_hairstyle(self, gender, face_shape, hair_type):
        """
        Recommends hairstyles based on gender, face shape, and hair type.

        Args:
            gender (str): 'male' or 'female'.
            face_shape (str): e.g., 'round', 'square', 'oval', 'heart'.
            hair_type (str): e.g., 'straight', 'wavy', 'curly', 'dreadlocks', 'kinky'.

        Returns:
            list: A list of dictionaries containing recommended hairstyles and their descriptions.
        """
        if self.df is None:
            return [{"style": "Unable to load recommendations", "description": "Data file not found"}]

        try:
            # Convert inputs to lowercase and map gender names
            gender = gender.lower()
            face_shape = face_shape.lower()
            hair_type = hair_type.lower()

            # Map gender from detector format (Men/Women) to recommender format (male/female)
            gender_mapping = {
                'men': 'male',
                'women': 'female'
            }
            gender = gender_mapping.get(gender, gender)

            # Map similar face shapes
            face_shape_mapping = {
                'oblong': 'oval',  # Oblong faces can use oval face recommendations
                'rectangular': 'square',  # Rectangular faces can use square face recommendations
                'diamond': 'heart',  # Diamond faces can use heart face recommendations
                'triangle': 'heart'  # Triangle faces can use heart face recommendations
            }

            # Map the face shape if it's one of the alternative shapes
            mapped_face_shape = face_shape_mapping.get(face_shape, face_shape)

            print(f"Looking for recommendations with: gender='{gender}', face_shape='{mapped_face_shape}', hair_type='{hair_type}'")
            
            # Filter recommendations based on input criteria
            matching_styles = self.df[
                (self.df['Gender'].str.lower() == gender) &
                (self.df['Face_Shape'].str.lower() == mapped_face_shape) &
                (self.df['Hair_Type'].str.lower() == hair_type)
            ]
            
            print(f"Found {len(matching_styles)} matching styles")

            if matching_styles.empty:
                print(f"No recommendations found for gender: {gender}, face shape: {face_shape} (mapped to {mapped_face_shape}), hair type: {hair_type}")
                return [{"style": "No specific recommendations found", 
                        "description": f"Original face shape '{face_shape}' was mapped to '{mapped_face_shape}' for recommendations"}]

            # Convert matches to list of dictionaries
            recommendations = []
            for _, row in matching_styles.iterrows():
                recommendations.append({
                    "style": row['Recommended_Style'],
                    "description": row['Description']
                })

            return recommendations

        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return [{"style": "Error getting recommendations", 
                    "description": "Please try again"}]

# Create a global instance of the recommender
recommender = HairstyleRecommender()

def recommend_hairstyle(gender, face_shape, hair_type):
    """
    Wrapper function for backward compatibility.
    
    Args:
        gender (str): 'male' or 'female'.
        face_shape (str): e.g., 'round', 'square', 'oval', 'heart'.
        hair_type (str): e.g., 'straight', 'wavy', 'curly', 'dreadlocks', 'kinky'.

    Returns:
        list: A list of recommended hairstyles with descriptions.
    """
    recommendations = recommender.recommend_hairstyle(gender, face_shape, hair_type)
    return [f"{rec['style']}: {rec['description']}" for rec in recommendations]
