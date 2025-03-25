import email
import os
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import Dict, List, Tuple

class LoanBankingEmailProcessor:
    def __init__(self):
        # Initialize without loading models
        self.zero_shot_classifier = None
        self.tokenizer = None
        self.model = None
        self.categories = {
            "Loan Application": ["New Application", "Documents Required", "Verification", "Approval Status"],
            "Loan Servicing": ["Payment Inquiry", "Statement Request", "Late Payment", "Prepayment"],
            "Customer Support": ["Account Inquiry", "Complaint", "Technical Issue", "General Query"],
            "Collections": ["Overdue Notice", "Payment Arrangement", "Default Warning", "Recovery"],
            "Marketing": ["Loan Offers", "Rate Updates", "Promotions", "Newsletters"]
        }
        self.label_map = {}
        idx = 0
        for cat, subcats in self.categories.items():
            for subcat in subcats:
                self.label_map[f"{cat}_{subcat}"] = idx
                idx += 1
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.email_history = []
        self.feedback_data = {'texts': [], 'labels': []}



    def _load_zero_shot(self):
        if not self.zero_shot_classifier:
            self.zero_shot_classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")


    def _load_distilbert(self):
        # Load DistilBERT only when needed for fine-tuning
        if not self.tokenizer or not self.model:
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=20)

    def categorize_email(self, email: Dict) -> Tuple[str, str, float]:
        text = f"{email['subject']} {email['content']}"
        if len(self.feedback_data['texts']) >= 10:  # Use fine-tuned model after feedback
            self._load_distilbert()
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[0][predicted_idx].item()
            for label, idx in self.label_map.items():
                if idx == predicted_idx:
                    category, subcategory = label.split('_')
                    return category, subcategory, confidence
        else:  # Use zero-shot classification
            self._load_zero_shot()
            candidate_labels = list(self.categories.keys())
            result = self.zero_shot_classifier(text, candidate_labels)
            category = result['labels'][0]
            subcategories = self.categories[category]
            sub_result = self.zero_shot_classifier(text, subcategories)
            subcategory = sub_result['labels'][0]
            confidence = (result['scores'][0] + sub_result['scores'][0]) / 2
            return category, subcategory, confidence

    def detect_duplicate(self, email: Dict, threshold: float = 0.9) -> bool:
        current_text = f"{email['subject']} {email['content']}"
        if not self.email_history:
            self.email_history.append(current_text)
            return False
        all_texts = self.email_history + [current_text]
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        similarity_matrix = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        max_similarity = np.max(similarity_matrix)
        if max_similarity >= threshold:
            return True
        self.email_history.append(current_text)
        return False

    def update_with_feedback(self, text: str, correct_category: str, correct_subcategory: str):
        label = f"{correct_category}_{correct_subcategory}"
        if label not in self.label_map:
            raise ValueError(f"Invalid category/subcategory combination: {label}")
        label_id = self.label_map[label]
        self.feedback_data['texts'].append(text)
        self.feedback_data['labels'].append(label_id)
        if len(self.feedback_data['texts']) >= 10:
            self._load_distilbert()
            self._fine_tune_model()

    def _fine_tune_model(self):
        encodings = self.tokenizer(self.feedback_data['texts'], truncation=True, padding=True, max_length=512, return_tensors='pt')
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(self.feedback_data['labels'])
        )
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Reduce batch size for memory
            warmup_steps=100,  # Fewer warmup steps
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )
        trainer = Trainer(model=self.model, args=training_args, train_dataset=dataset)
        trainer.train()

    def process_emails(self, folder_path: str) -> pd.DataFrame:
        emails = self.read_emails_from_folder(folder_path)
        results = []
        for email in emails:
            category, subcategory, confidence = self.categorize_email(email)
            is_duplicate = self.detect_duplicate(email)
            results.append({
                'filename': email['filename'],
                'subject': email['subject'],
                'from': email['from'],
                'category': category,
                'subcategory': subcategory,
                'confidence': f"{confidence:.2%}",
                'is_duplicate': is_duplicate
            })
        return pd.DataFrame(results)

    def read_emails_from_folder(self, folder_path: str) -> List[Dict]:
        emails = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.eml'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    msg = email.message_from_string(file.read())
                    email_data = {
                        'subject': msg.get('Subject', ''),
                        'from': msg.get('From', ''),
                        'content': self._get_email_content(msg),
                        'filename': filename
                    }
                    emails.append(email_data)
        return emails

    def _get_email_content(self, msg) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    return part.get_payload(decode=True).decode('utf-8', errors='ignore')
        else:
            return msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        return ""
