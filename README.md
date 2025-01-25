
# News Verification App

## Overview

This application is a **Fake News Detection System** built using **AI and Machine Learning**. The app allows users to validate the authenticity of news articles by predicting whether the content is **reliable** or **unreliable**. The model leverages **RoBERTa** (a transformer-based language model) for **text classification** to detect fake news. The app stores predictions and allows users to provide feedback, which is used to fine-tune the model for better accuracy over time.

## Features

- **Text Validation**: Users can input a news article to check whether it is reliable or not.
- **User Feedback**: Users can provide feedback (correct or wrong) for each prediction, which helps in improving the model.
- **Database**: Stores the validated news articles along with their scores and conclusions.
- **Model Update**: The model can be updated using new user feedback, allowing continuous improvement.
- **Web Interface**: Simple and clean UI for user interactions.

## Installation

### Requirements
- Python 3.7+
- Flask
- SQLAlchemy
- Transformers
- PyTorch

You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Setting Up the Database

To set up the SQLite database, run the following commands:

```bash
python
>>> from app import db
>>> db.create_all()
```

This will create the required database tables.

## Usage

1. **Running the App**: 

   To run the app locally, use the following command:

   ```bash
   python app.py
   ```

2. **Accessing the App**:

   After running the app, open your browser and go to:

   ```
   http://127.0.0.1:5000
   ```

3. **Validate News**:

   - Go to the "Validate Text" page to input the text of the news article you want to check.
   - The app will return a conclusion (reliable or unreliable) along with scores for fake and real news.

4. **Provide Feedback**:

   - After validating the news article, you can provide feedback (correct or wrong).
   - If the prediction was incorrect, the model will be updated with the feedback to improve accuracy.

## How It Works

- **Model**: The core of the application is the RoBERTa-based model (`hamzab/roberta-fake-news-classification`) which is used to classify news articles.
  
- **Text Classification**: The input text is tokenized and passed through the model, which predicts whether the news is fake or real. The result is displayed to the user, along with a confidence score.

- **Model Update**: Feedback provided by users helps in retraining the model with new data, improving its predictions over time.

## Model

This app uses the **RoBERTa** model from Hugging Face’s model hub (`hamzab/roberta-fake-news-classification`).

## Contributing

If you'd like to contribute to this project, feel free to fork the repository, make your changes, and submit a pull request.

### Steps to Contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Test your changes.
5. Submit a pull request with a clear description of your modifications.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Hugging Face**: For providing pre-trained models like RoBERTa.
- **PyTorch**: For the deep learning framework used for model training and inference.
- **Flask**: For the web framework to build this application.
