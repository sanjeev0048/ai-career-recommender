import json
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Career Role Recommender 🤖💼")
st.write("Enter your skills to discover best job roles!")

import os
roles_path = os.path.join(os.path.dirname(__file__), "roles.json")
with open(roles_path, "r") as f:
    roles_data = json.load(f)


# Combine skills for each role into a single string
roles_list = [" ".join(role["skills"]) for role in roles_data]

# Create TF-IDF vectors for all roles
vectorizer = TfidfVectorizer()
role_vectors = vectorizer.fit_transform(roles_list)

skills = st.text_area("Enter your skills (comma separated):")

if st.button("Recommend Roles"):
    if not skills.strip():
        st.warning("Please enter some skills!")
    else:
        # Vectorize user skills
        user_vector = vectorizer.transform([skills.lower()])
        similarities = cosine_similarity(user_vector, role_vectors)[0]

        # Pair each role with its similarity score
        ranked = sorted(
            zip(roles_data, similarities),
            key=lambda x: x[1],
            reverse=True
        )

        st.subheader("🔥 Top Matching Job Roles")
        for role, score in ranked[:3]:
            st.markdown(f"### {role['role']} — **{round(score * 100, 2)}% Match**")
            st.write(f"📌 Roadmap: {role['roadmap']}")
            st.write("---")
