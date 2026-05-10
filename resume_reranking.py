# ==========================================
# Resume Reranking System
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# Job Description
# ==========================================

job_description = """
Python developer with machine learning,
deep learning, NLP, and data analysis skills.
"""

# ==========================================
# Sample Resume Data
# ==========================================

resumes = [

    "Experienced Python developer with NLP and machine learning knowledge",

    "Java developer with frontend and backend experience",

    "Data scientist skilled in deep learning and Python",

    "Graphic designer with Photoshop expertise"
]

candidate_names = [

    "Candidate 1",

    "Candidate 2",

    "Candidate 3",

    "Candidate 4"
]

print("\nResume Reranking System Started!")

# ==========================================
# TF-IDF Vectorization
# ==========================================

vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(
    [job_description] + resumes
)

print("\nVectorization Completed!")

# ==========================================
# Cosine Similarity
# ==========================================

similarity_scores = cosine_similarity(
    vectors[0:1],
    vectors[1:]
).flatten()

# ==========================================
# Ranking Candidates
# ==========================================

ranking = sorted(
    zip(candidate_names, similarity_scores),
    key=lambda x: x[1],
    reverse=True
)

print("\nCandidate Ranking:\n")

for name, score in ranking:

    print(
        name,
        "-> Similarity Score:",
        round(score, 2)
    )

# ==========================================
# Visualization
# ==========================================

names = [x[0] for x in ranking]

scores = [x[1] for x in ranking]

plt.figure(figsize=(8,5))

plt.bar(names, scores)

plt.title("Resume Similarity Scores")

plt.xlabel("Candidates")

plt.ylabel("Similarity Score")

plt.show()