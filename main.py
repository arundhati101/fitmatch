import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
df = pd.read_csv('megaGymDataset.csv')

# Data Preprocessing
df['Desc'] = df['Desc'].fillna('')  # Fill NaN values in 'Desc' with empty strings

# Create the 'features' column, handling potential NaN values in other columns
df['features'] = (df['Title'].fillna('') + ' ' +
                   df['Desc'] + ' ' +
                   df['Type'].fillna('') + ' ' +
                   df['BodyPart'].fillna('') + ' ' +
                   df['Equipment'].fillna('') + ' ' +
                   df['Level'].fillna(''))

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['features'])

# Recommendation Function
def recommend_exercises(goal, body_type, fitness_level, equipment, body_part, top_n=5):
    """
    Recommend exercises based on user preferences.
    """
    user_preferences = f"{goal} {body_type} {fitness_level} {equipment} {body_part}"
    user_vector = tfidf_vectorizer.transform([user_preferences])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    recommendations = df.iloc[top_indices][['Title', 'Type', 'BodyPart', 'Equipment', 'Level', 'Desc']]
    return recommendations

# Example User Inputs
goal = "Muscle Gain"
body_type = "Mesomorph"
fitness_level = "Intermediate"
equipment = "Dumbbells"
body_part = "Chest"

# Get Recommendations
recommendations = recommend_exercises(goal, body_type, fitness_level, equipment, body_part)

# Output
print("\nRecommended Exercises:\n")
for i, row in recommendations.iterrows():
    print(f"Exercise {i + 1}:")
    print(f"  Title       : {row['Title']}")
    print(f"  Type        : {row['Type']}")
    print(f"  Body Part   : {row['BodyPart']}")
    print(f"  Equipment   : {row['Equipment']}")
    print(f"  Level       : {row['Level']}")
    print(f"  Description : {row['Desc']}")
    print("-" * 50)
