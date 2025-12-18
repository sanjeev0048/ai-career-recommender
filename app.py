import json
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# App Title
st.title("AI Career Role Recommender 🤖💼")
st.write("Enter your skills and get personalized role suggestions!")

# Load roles data
with open("roles.json", "r") as f:
    roles_data = json.load(f)

# Prepare role texts and names
role_texts = [" ".join(role["skills"]) for role in roles_data]
role_names = [role["role"] for role in roles_data]

# Vectorization
vectorizer = TfidfVectorizer()
role_vectors = vectorizer.fit_transform(role_texts)

# User input
user_input = st.text_area("Enter your skills (comma separated):", height=100)

# Recommend button
if st.button("Recommend Roles"):
    if not user_input.strip():
        st.warning("Please enter some skills!")
    else:
        # Transform user input
        user_vec = vectorizer.transform([user_input.lower()])
        similarities = cosine_similarity(user_vec, role_vectors)[0]

        # Rank all roles
        ranked = sorted(
            zip(role_names, similarities, roles_data),
            key=lambda x: x[1],
            reverse=True
        )

        st.subheader("Career Matches 🎯")

        # Show ALL roles
        for role, score, role_info in ranked:
            st.markdown(f"### {role}")
            st.write(f"Match Score: **{round(score * 100, 2)}%**")
            st.write(f"🛠 Roadmap: {role_info['roadmap']}")
            st.markdown("---")
