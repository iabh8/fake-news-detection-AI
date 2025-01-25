import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import logging

logging.basicConfig(level=logging.DEBUG)

model_name_or_path = 'hamzab/roberta-fake-news-classification'
tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
model = RobertaForSequenceClassification.from_pretrained(model_name_or_path)


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///news_verification.db'
app.config['SECRET_KEY'] = 'your_secret_key_here'
db = SQLAlchemy(app)


class NewsArticle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    score_label_1 = db.Column(db.String(64), nullable=False)
    score_label_0 = db.Column(db.String(64), nullable=False)
    conclusion = db.Column(db.String(64), nullable=False)
    user_feedback = db.Column(db.String(64), nullable=True)


with app.app_context():
    db.create_all()

# Function to preprocess data and predict fake or real news
def predict_fake_news(text):
    inputs = tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        logging.debug(f"Logits: {logits}")
        
        scores = torch.softmax(logits, dim=-1).tolist()[0]
        logging.debug(f"Softmax scores: {scores}")
        
        score_label_1 = format(scores[1], '.16f')  # Real news
        score_label_0 = format(scores[0], '.16f')  # Fake news
        
    # Determine the label with the higher score
    if scores[1] > scores[0]:
        conclusion = "Reliable"
    else:
        conclusion = "Unreliable"
    
    result = {
        "score_label_1": score_label_1,
        "score_label_0": score_label_0,
        "conclusion": conclusion
    }
    
    return result

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Function to update model with new data
def update_model(new_texts, new_labels, model, tokenizer, epochs=1, batch_size=8, learning_rate=5e-5):
    dataset = NewsDataset(new_texts, new_labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

    model.save_pretrained('./MODEL')
    tokenizer.save_pretrained('./MODEL')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/validate_text', methods=['GET', 'POST'])
def validate_text():
    if request.method == 'POST':
        news_text = request.form['news_text']
        result = predict_fake_news(news_text)
        
        
        new_article = NewsArticle(
            text=news_text,
            score_label_1=result['score_label_1'],
            score_label_0=result['score_label_0'],
            conclusion=result['conclusion']
        )
        db.session.add(new_article)
        db.session.commit()

        # Add the article ID to the result dictionary
        result['article_id'] = new_article.id

        return render_template('validate_text.html', result=result, news_text=news_text)
    
    return render_template('validate_text.html', result=None, news_text=None)

@app.route('/feedback/<int:article_id>', methods=['POST'])
def feedback(article_id):
    feedback = request.form['feedback']
    article = NewsArticle.query.get(article_id)
    if article:
        article.user_feedback = feedback
        db.session.commit()
        
        
        if feedback == 'wrong':
            new_texts = [article.text]
            new_labels = [1 if article.conclusion == "Unreliable" else 0]
            update_model(new_texts, new_labels, model, tokenizer)
            flash('Thank you for your feedback! The model has been updated.', 'success')
        else:
            flash('Thank you for your feedback!', 'success')
    
    return redirect(url_for('validate_text'))

if __name__ == "__main__":
    app.run(debug=True)
