# AI Career Role Recommender 🤖💼

A smart web app that recommends the best job roles based on your listed technical skills.  
Built with Python, Streamlit, and Scikit-Learn — lightweight, fast, and deploy-ready! 💪

---

### 🔥 Features
- Enter skills (comma-separated)
- Calculates similarity with job roles using TF-IDF + cosine similarity
- Shows **Top 3 matching roles** with match percentage
- Provides a learning roadmap for career improvement
- Fully extendable — skill & role data stored in JSON

---

### 🛠️ Tech Stack
| Category | Technology |
|---------|------------|
| Language | Python |
| Web Framework | Streamlit |
| Machine Learning | Scikit-Learn |
| Data Store | JSON |

---

### 🚀 Run the App Locally

```bash
git clone https://github.com/sanjeev0048/ai-career-recommender.git
cd ai-career-recommender

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
