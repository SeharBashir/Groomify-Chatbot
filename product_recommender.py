import pandas as pd
import numpy as np

class ProductRecommender:
    def __init__(self, cosmetics_csv_path):
        self.df = pd.read_csv(cosmetics_csv_path)
        
    def recommend_products(self, skin_type, top_n=5):
        """
        Recommend products based on skin type
        Args:
            skin_type: str, one of 'dry', 'normal', 'oily'
            top_n: int, number of products to recommend
        Returns:
            list of dict containing product recommendations
        """
        try:
            # Convert skin type to title case to match CSV
            skin_type = skin_type.title()
            
            # Filter products suitable for the skin type (score > 0)
            suitable_products = self.df[self.df[skin_type] > 0].copy()
            
            # Sort by rank (higher is better) and price (lower is better)
            suitable_products['score'] = suitable_products['Rank'] - suitable_products['Price'] / 1000
            sorted_products = suitable_products.sort_values('score', ascending=False)
            
            # Get top N products
            top_products = sorted_products.head(top_n)
            
            recommendations = []
            for _, product in top_products.iterrows():
                recommendations.append({
                    'label': product['Label'],
                    'brand': product['Brand'],
                    'name': product['Name'],
                    'price': f"${product['Price']}",
                    'rating': product['Rank'],
                    'key_ingredients': ", ".join(product['Ingredients'].split(", ")[:3]) + "..."
                })
            
            return recommendations
            
        except Exception as e:
            print(f"Error recommending products: {str(e)}")
            return []
