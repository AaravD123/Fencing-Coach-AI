import json

# Sample manually written questions and answers (you can expand later)
q_and_a = [
    {
        "question": "What is the minimum age to compete in Division I?",
        "answer": "Fencers must be at least 13 years old as of January 1 to compete in Division I events."
    },
    {
        "question": "How can a fencer earn a classification?",
        "answer": "By placing in the top percentage of certain classified events."
    },
    {
        "question": "What equipment is required at competitions?",
        "answer": "Fencers need a regulation mask, jacket, glove, underarm protector, and appropriate weapon."
    },
    {
        "question": "Can I represent a club outside of the USA?",
        "answer": "No. Fencers may only represent clubs registered with USA Fencing."
    },
    {
        "question": "How long is a rating valid?",
        "answer": "For four seasons unless renewed by performance."
    }
]

# Save the file
with open("../data/fencing_qna_dataset.json", "w", encoding="utf-8") as f:
    json.dump({"q_and_a": q_and_a}, f, indent=2)

print("Saved Q&A dataset to data/fencing_qna_dataset.json")
