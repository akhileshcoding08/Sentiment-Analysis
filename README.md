# Movie Review Sentiment Analyzer — IMDB Dataset + Flask Deployment

An end-to-end sentiment analysis project that classifies movie reviews as **Positive** or **Negative**. It is trained on a real, cleaned, deduplicated 15,000-review subset of the Stanford IMDB Large Movie Review Dataset, and deployed as an interactive web app using **Flask**, **HTML**, and **CSS**.

## 📌 Overview

This project covers the complete machine learning lifecycle:

1. **Dataset** — a real, balanced, deduplicated 15,000-review dataset sourced from the Stanford AI Large Movie Review Dataset (Maas et al., ACL 2011).
2. **Model development** — a TF-IDF + Logistic Regression pipeline trained and evaluated in a Jupyter notebook, achieving **~87% test accuracy** on genuine held-out data.
3. **Model deployment** — the trained pipeline is saved as a pickle (`.pkl`) file.
4. **Web application** — a Flask app with a styled HTML/CSS front end where users type a review and get a live sentiment prediction with a confidence score.

> Note: this repository also contains an earlier exploratory notebook (`Sentiment_Analysis.ipynb`) that builds and trains a **Bidirectional LSTM** on the Keras built-in IMDB dataset. The **final, deployed** model (`Sentiment_Analysis_Final.ipynb`) uses TF-IDF + Logistic Regression on the real 15,000-review dataset described above, since it trains fast, requires no TensorFlow at serving time, and pickles reliably for Flask.

## 🚀 Features

- Real, deduplicated, class-balanced dataset (7,500 positive / 7,500 negative reviews)
- Text cleaning pipeline (HTML tag removal, punctuation stripping, lowercasing)
- TF-IDF vectorization (unigrams + bigrams, 8,000 features)
- Logistic Regression classifier — fast to train, reliably picklable
- Full evaluation: accuracy, precision/recall/F1, confusion matrix
- Deployment-ready `.pkl` artifact
- Flask web app with a form-based UI and a JSON `/predict` API
- Attractive, responsive, gradient-styled HTML/CSS front end with a live confidence bar

## 🛠️ Tech Stack

- **Python 3**
- **scikit-learn** — TF-IDF vectorizer + Logistic Regression
- **Pandas / NumPy** — data handling
- **Matplotlib** — confusion matrix visualization
- **Flask** — web app / API
- **HTML5 / CSS3** — front-end interface
- **Jupyter Notebook** — development & experimentation
- **TensorFlow / Keras** — used in the earlier exploratory Bidirectional LSTM notebook

## 📂 Project Structure

```
Sentiment-Analysis/
│
├── Sentiment_Analysis_Final.ipynb    # Final notebook: real dataset → training → pickle
├── Sentiment_Analysis.ipynb          # Earlier exploratory Bidirectional LSTM notebook
├── train_final_model.py              # Standalone script version of the training pipeline
├── app.py                            # Flask application (routes: "/" and "/predict")
├── sentiment_model.pkl               # Pickled TF-IDF vectorizer + Logistic Regression model
├── templates/
│   └── index.html                    # Styled front-end page (HTML + CSS)
├── imdb_sentiment_dataset_15000.csv  # Real, cleaned, balanced 15,000-review dataset
├── DATASET_README.md                 # Dataset source, cleaning steps, and citation
├── README.md                         # Project documentation
└── requirements.txt                  # Python dependencies
```

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Sentiment-Analysis.git
   cd Sentiment-Analysis
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

### 1. Explore / retrain the model
```bash
jupyter notebook Sentiment_Analysis_Final.ipynb
```
Run all cells to load the dataset, clean the text, vectorize it, train the Logistic Regression model, evaluate it, and regenerate `sentiment_model.pkl`.

Alternatively, retrain from the command line:
```bash
python train_final_model.py
```

### 2. Run the web app
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser, type a movie review, and click **Analyze Sentiment** to see the predicted sentiment and confidence score.

### 3. Use the JSON API
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was absolutely brilliant!"}'
```
Response:
```json
{"sentiment": "Positive", "confidence": 96.42}
```

## 📦 Dataset

- **Records:** 15,000 (7,500 positive / 7,500 negative — perfectly balanced)
- **Source:** Stanford AI Large Movie Review Dataset (Maas et al., ACL 2011), via the GitHub mirror `laxmimerit/IMDB-Movie-Reviews-Large-Dataset-50k`
- **Cleaning:** merged train + test splits, removed 25,096 exact duplicate reviews, filtered near-empty entries, then sampled a balanced subset
- Full details, citation, and column descriptions are in [`DATASET_README.md`](DATASET_README.md)

## 🧠 Model Pipeline (Final / Deployed Model)

| Step | Details |
|---|---|
| Text cleaning | Remove HTML tags, punctuation; lowercase |
| Vectorization | TF-IDF, unigrams + bigrams, max 8,000 features, English stop words removed |
| Model | Logistic Regression (max_iter=1000, C=1.0) |
| Train/test split | 80% / 20%, stratified by sentiment |

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | **87.17%** |
| Precision (avg) | 0.87 |
| Recall (avg) | 0.87 |
| F1-score (avg) | 0.87 |
| Training samples | 12,000 |
| Testing samples | 3,000 |

## 🔮 Sample Predictions

| Review Text | Predicted Sentiment |
|---|---|
| "This film was absolutely brilliant, I loved every scene!" | Positive |
| "What a boring and terrible waste of time." | Negative |
| "The acting was decent but the story dragged on forever." | Negative |

## 📈 Future Improvements

- Train and serve the Bidirectional LSTM directly in Flask (`tensorflow.keras.models.load_model()`) for potentially higher accuracy
- Use pre-trained word embeddings (GloVe/Word2Vec) or transformer-based models (BERT, DistilBERT)
- Expand the dataset beyond 15,000 reviews (up to the full ~24,900 unique reviews available)
- Add Docker support for containerized deployment
- Deploy to a cloud platform (Render, Railway, AWS, Azure, or Heroku)
- Add input validation, rate limiting, and logging to the Flask API

## 📄 License

This project is open source and available under the [MIT License](LICENSE). The dataset is derived from the Stanford AI Large Movie Review Dataset — please cite Maas et al. (2011) if you publish work using it (see `DATASET_README.md`).

## 🙋 Author

Feel free to reach out or open an issue for suggestions and improvements.
