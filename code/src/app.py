# app.py
from flask import Flask, request, jsonify
from loan_banking_processor import LoanBankingEmailProcessor

app = Flask(__name__)
processor = LoanBankingEmailProcessor()

@app.route('/process_emails', methods=['POST'])
def process_emails():
    emails = request.json.get('emails', [])  # Expect list of email dicts
    results = []
    for email in emails:
        category, subcategory, confidence = processor.categorize_email(email)
        is_duplicate = processor.detect_duplicate(email)
        results.append({
            'subject': email['subject'],
            'from': email['from'],
            'category': category,
            'subcategory': subcategory,
            'confidence': f"{confidence:.2%}",
            'is_duplicate': is_duplicate
        })
    return jsonify(results)

@app.route('/feedback', methods=['POST'])
def provide_feedback():
    data = request.json
    text = data.get('text')
    category = data.get('category')
    subcategory = data.get('subcategory')
    processor.update_with_feedback(text, category, subcategory)
    return jsonify({"message": "Feedback processed successfully"})

if __name__ == '__main__':
    app.run()