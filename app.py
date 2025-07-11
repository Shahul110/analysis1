from flask import Flask, render_template
#from body_module  import run_body_analysis   # ← your function (returns dict)
from cloth_module import run_cloth_analysis  # ← must also return dict

app = Flask(__name__)
@app.route("/")
def home():
    return render_template("home.html")




@app.route("/analyze")
def analyze():
    # ① body analysis (blocks until finished)
    body_scores = run_cloth_analysis()          # returns dict

    # ② cloth analysis (blocks until finished)
    cloth_scores = run_cloth_analysis()        # returns dict

    # ③ combine 50‑50
    combined = {}
    for p in set(body_scores) | set(cloth_scores):
        combined[p] = 0.5 * body_scores.get(p, 0) + 0.5 * cloth_scores.get(p, 0)

    final_persona = max(combined, key=combined.get)
    final_score   = combined[final_persona]

    return render_template(
        "result.html",
        combined=combined,
        final_persona=final_persona,
        final_score=f"{final_score:.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)

'''
#body movement
body_scores = run_body_analysis()

#cloth
cloth_scores = run_cloth_analysis()
combined_scores = {}
for persona in set(body_scores.keys()).union(cloth_scores.keys()):
    body_score = body_scores.get(persona, 0.0)
    cloth_score = cloth_scores.get(persona, 0.0)
    combined_scores[persona] = 0.5 * body_score + 0.5 * cloth_score

final_persona = max(combined_scores, key=combined_scores.get)
final_score = combined_scores[final_persona]

print("\n🎯 Combined Persona Scores:")
for persona, score in combined_scores.items():
    print(f"{persona}: {score:.2f}")

print(f"\n✅ Final Persona: {final_persona} (Confidence Score: {final_score:.2f})")
'''
