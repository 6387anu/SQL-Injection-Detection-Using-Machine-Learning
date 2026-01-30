import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
import joblib
import shap
import io
import base64
import pandas as pd
from scipy.sparse import issparse

app = Flask(__name__)

# --- Load Models ---
try:
    model = joblib.load('xgb_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    explainer = shap.TreeExplainer(model)
except Exception as e:
    print(f"Error loading models: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Model not loaded", 500

    query_text = request.form['query']
    
    # 1. Prediction
    transformed_query = vectorizer.transform([query_text])
    prediction = model.predict(transformed_query)[0]
    prediction_proba = model.predict_proba(transformed_query)[0][1]

    # 2. SHAP Values
    shap_values = explainer.shap_values(transformed_query)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if issparse(shap_values):
        shap_values = shap_values.toarray()

    # 3. Generate Data for UI
    feature_names = vectorizer.get_feature_names_out()
    df = pd.DataFrame({'ngram': feature_names, 'shap_value': shap_values[0]})
    
    # Sort for the Table
    df['abs_shap'] = df['shap_value'].abs()
    df_sorted = df.sort_values(by='abs_shap', ascending=False).head(10)
    top_features = df_sorted[['ngram', 'shap_value']].to_dict(orient='records')

    # 4. Generate the "Text Explanation" (New Feature)
    if prediction == 1:
        # Filter for positive (bad) features only
        bad_patterns = df[df['shap_value'] > 0].sort_values(by='shap_value', ascending=False).head(3)
        patterns_list = [f"'{row['ngram']}'" for _, row in bad_patterns.iterrows()]
        
        if patterns_list:
            patterns_str = ", ".join(patterns_list)
            explanation_text = (f"This query was flagged as MALICIOUS because the ML detected suspicious "
                                f"patterns commonly found in SQL injection attacks, specifically: {patterns_str}.")
        else:
            explanation_text = "This query was flagged as MALICIOUS due to a combination of subtle character patterns."
        
        result_text = "MALICIOUS (SQL Injection Detected)"
        css_class = "danger"
    else:
        explanation_text = "This query appears safe. No significant malicious patterns were detected."
        result_text = "SAFE (Normal Query)"
        css_class = "success"

    # 5. Generate Plot
    plt.figure(figsize=(10, 5))
    shap.plots.bar(shap.Explanation(values=shap_values[0], 
                                    base_values=explainer.expected_value, 
                                    data=None, 
                                    feature_names=feature_names),
                   max_display=10, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')

    return render_template('index.html', 
                           prediction_text=result_text, 
                           query=query_text, 
                           css_class=css_class,
                           plot_url=plot_url,
                           top_features=top_features,
                           probability=round(prediction_proba * 100, 2),
                           explanation_text=explanation_text) # Sending the new text

if __name__ == '__main__':
    app.run(debug=True, port=5000)