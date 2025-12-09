import json
import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.title("AI Career Role Recommender 🤖💼")
st.write("Enter your skills and get personalized role suggestions!")

model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

with open("roles.json", "r") as f:
    roles_data = json.load(f)

user_input = st.text_area("Enter your skills (comma separated):", height=100)

if st.button("Recommend Roles"):
    if user_input.strip() == "":
        st.warning("Please enter some skills!")
    else:
        user_vec = model.encode(user_input.lower(), convert_to_tensor=True)
        
        ranked = []
        for role in roles_data:
            role_vec = model.encode(" ".join(role["skills"]), convert_to_tensor=True)
            score = util.cos_sim(user_vec, role_vec).item()
            ranked.append((role["role"], score, role["roadmap"]))

        ranked.sort(key=lambda x: x[1], reverse=True)

        st.subheader("Top Role Matches:")
        for role, score, roadmap in ranked[:3]:
            st.markdown(f"### {role}")
            st.write(f"Match Score: **{round(score*100, 2)}%**")
            st.write(f"📌 Roadmap: {roadmap}")
            st.markdown("---")
