# Sentiment Analysis with Custom Negation Handling
*A unique NLP project using logistic regression and TF-IDF*

## 📌 Description
This project performs sentiment analysis on movie reviews using logistic regression. Unlike typical models, it includes **custom negation handling** to improve accuracy when phrases like “not good” or “didn’t enjoy” are used. This small but powerful change leads to more accurate sentiment predictions.

## 🚀 Key Features
- Preprocessing (HTML tags, URLs, punctuations removal)
- TF-IDF vectorization with top 5000 features
- Logistic Regression classifier
- Custom handling of negations (e.g., `not good` → `not_good`)
- Accuracy comparison: with vs. without negation handling
- Custom test predictions to showcase difference

## 🧠 Uniqueness
Traditional sentiment analysis often fails to capture the impact of negation. This project introduces a **manual negation tagging mechanism**, allowing models to better understand phrases like “not happy,” “never liked,” or “don’t recommend.”

## 📂 Files
- `sentiment_with_negation.py`: Main project with custom logic
- `sentiment_normal.py`: Baseline model without negation handling
- `README.md`: Documentation
- `IMDB Dataset.csv`: Dataset used for training/testing. Not included due to licensing restrictions. You can download it from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) or other open sources.
- `output.txt`: Contains comparison of normal vs. custom sentiment outputs, demonstrating negation handling improvements.

## License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for full license details

## 🔧 How to Run
```bash
# Activate virtual environment
.\venv\Scripts\activate

# Install requirements (if you create requirements.txt)
pip install -r requirements.txt

# Run the model
python sentiment_with_negation.py
```
