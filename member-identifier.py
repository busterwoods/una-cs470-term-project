from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import FeatureUnion, Pipeline
from wordcloud import WordCloud

DATA_FILE = Path(__file__).with_name("messages.csv")


class RobertaTokenizerVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        analyzer="word",
        ngram_range=(1, 1),
        min_df=1,
        max_df=1.0,
        sublinear_tf=False,
        strip_accents=None,
        tokenizer_name="roberta-base",
    ):
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.strip_accents = strip_accents
        self.tokenizer_name = tokenizer_name

    def _load_tokenizer(self):
        try:
            from transformers import RobertaTokenizerFast
        except ImportError as error:
            raise ImportError(
                "RobertaTokenizerVectorizer requires the 'transformers' package."
            ) from error

        return RobertaTokenizerFast.from_pretrained(self.tokenizer_name)

    def _tokenize(self, raw_documents):
        return [
            self.tokenizer(
                str(document),
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            for document in raw_documents
        ]

    def _document_ngrams(self, token_ids):
        min_n, max_n = self.ngram_range
        if min_n < 1 or min_n > max_n:
            raise ValueError("ngram_range must contain positive values in ascending order.")

        return [
            tuple(token_ids[index:index + size])
            for size in range(min_n, max_n + 1)
            for index in range(len(token_ids) - size + 1)
        ]

    def fit(self, raw_documents, y=None):
        self.tokenizer = self._load_tokenizer()
        tokenized_documents = self._tokenize(raw_documents)
        document_ngrams = [
            self._document_ngrams(token_ids)
            for token_ids in tokenized_documents
        ]

        document_count = len(document_ngrams)
        if document_count == 0:
            raise ValueError("Cannot fit RobertaTokenizerVectorizer on no documents.")

        min_documents = (
            int(np.ceil(self.min_df * document_count))
            if isinstance(self.min_df, float)
            else self.min_df
        )
        max_documents = (
            int(np.floor(self.max_df * document_count))
            if isinstance(self.max_df, float)
            else self.max_df
        )
        document_frequencies = {}
        for ngrams in document_ngrams:
            for ngram in set(ngrams):
                document_frequencies[ngram] = document_frequencies.get(ngram, 0) + 1

        selected_ngrams = [
            ngram
            for ngram, frequency in document_frequencies.items()
            if frequency >= min_documents and frequency <= max_documents
        ]
        if not selected_ngrams:
            raise ValueError(
                "No RoBERTa token features remain after min_df/max_df filtering."
            )

        self.vocabulary_ = {
            ngram: index for index, ngram in enumerate(selected_ngrams)
        }
        counts = self._counts(document_ngrams)
        self.weighting_ = TfidfTransformer(
            sublinear_tf=self.sublinear_tf,
        ).fit(counts)
        return self

    def _counts(self, document_ngrams):
        rows = []
        columns = []
        values = []
        for row, ngrams in enumerate(document_ngrams):
            counts = {}
            for ngram in ngrams:
                column = self.vocabulary_.get(ngram)
                if column is not None:
                    counts[column] = counts.get(column, 0) + 1
            for column, value in counts.items():
                rows.append(row)
                columns.append(column)
                values.append(value)

        return csr_matrix(
            (values, (rows, columns)),
            shape=(len(document_ngrams), len(self.vocabulary_)),
            dtype=np.float64,
        )

    def transform(self, raw_documents):
        if not hasattr(self, "vocabulary_"):
            raise RuntimeError("RobertaTokenizerVectorizer must be fitted before transform.")

        tokenized_documents = self._tokenize(raw_documents)
        document_ngrams = [
            self._document_ngrams(token_ids)
            for token_ids in tokenized_documents
        ]
        return self.weighting_.transform(self._counts(document_ngrams))


def load_data():
    df = pd.read_csv(
        DATA_FILE,
        encoding="cp1252",
        encoding_errors="ignore",
        on_bad_lines="skip",
    )
    x = df["Content"].fillna("").astype(str).values
    y = df["Author"].values
    return x, y


def create_model1():
    features = FeatureUnion([
        ("word", RobertaTokenizerVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("subword", RobertaTokenizerVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
        )),
    ])

    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=1.0,
        )),
    ])


def create_cross_validation(y):
    smallest_class_size = pd.Series(y).value_counts().min()
    n_splits = min(5, smallest_class_size)
    if n_splits < 2:
        raise ValueError(
            "Each author must have at least two examples for cross-validation."
        )

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )


def plot_data(x, y):
    plot_features = RobertaTokenizerVectorizer(
        sublinear_tf=True,
        strip_accents="unicode",
    )
    plot_matrix = plot_features.fit_transform(x)
    roberta_2d = TruncatedSVD(
        n_components=2,
        random_state=42,
    ).fit_transform(plot_matrix)
    author_codes, authors = pd.factorize(y)

    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    scatter = axes[0].scatter(
        roberta_2d[:, 0],
        roberta_2d[:, 1],
        c=author_codes,
        cmap="tab10",
        alpha=0.75,
    )
    axes[0].set_xlabel("SVD component 1")
    axes[0].set_ylabel("SVD component 2")
    axes[0].set_title("RoBERTa Tokenizer Feature Scatterplot")
    axes[0].legend(
        handles=scatter.legend_elements()[0],
        labels=authors.tolist(),
        title="Author",
    )

    wordcloud = WordCloud(
        width=800,
        height=600,
        background_color="white",
    ).generate(" ".join(str(document) for document in x))
    axes[1].imshow(wordcloud, interpolation="bilinear")
    axes[1].axis("off")
    axes[1].set_title("Message Word Cloud")

    figure.tight_layout()
    plt.show()


def test_algorithms():
    x, y = load_data()
    cv = create_cross_validation(y)
    model1 = create_model1()
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": make_scorer(f1_score, average="macro"),
    }

    results1 = cross_validate(
        model1,
        x,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    print(f"Mean Accuracy (Logistic Regression): {results1['test_accuracy'].mean()}")
    print(f"Mean F1 Score (Logistic Regression): {results1['test_f1_macro'].mean()}")

    labels = np.unique(y)
    predictions = cross_val_predict(model1, x, y, cv=cv, n_jobs=-1)
    matrix = confusion_matrix(y, predictions, labels=labels)
    print("\nConfusion Matrix (Logistic Regression)")
    print(pd.DataFrame(matrix, index=labels, columns=labels))

    plt.figure(figsize=(8, 5))
    plt.plot(results1["test_accuracy"], label="Logistic Regression")
    plt.legend(title="Model")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Cross-Validation Scores")
    plt.show()

    model1.fit(x, y)
    return model1


def get_top_predictions(model, phrase, limit=3):
    probabilities = model.predict_proba([phrase])[0]
    ranked_indices = np.argsort(probabilities)[::-1][:limit]
    return [
        (model.classes_[index], probabilities[index] * 100)
        for index in ranked_indices
    ]


def format_predictions(predictions):
    return "\n".join(
        f"{rank}. {author} ({confidence:.2f}% confidence)"
        for rank, (author, confidence) in enumerate(predictions, start=1)
    )


def guess(model1):
    test_phrase = ""
    while test_phrase != "STOP":
        test_phrase = input("Enter text to identify: ").strip()
        if not test_phrase:
            print("No text was entered.")
            continue
        if test_phrase == "STOP":
            break

        test_phrase = test_phrase.replace("\n", " ")
        predictions = get_top_predictions(model1, test_phrase)
        print("Top predicted authors:")
        print(format_predictions(predictions))


def run_gui():
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("Message Identifier")
    root.geometry("620x380")
    root.minsize(500, 300)

    main_frame = ttk.Frame(root, padding=24)
    main_frame.pack(fill="both", expand=True)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(2, weight=1)

    ttk.Label(
        main_frame,
        text="Message Identifier",
        font=("TkDefaultFont", 18, "bold"),
    ).grid(row=0, column=0, sticky="w")
    status_label = ttk.Label(
        main_frame,
        text="Training the prediction model...",
    )
    status_label.grid(row=1, column=0, pady=(8, 12), sticky="w")

    ttk.Label(
        main_frame,
        text="Enter a phrase to identify:",
    ).grid(row=2, column=0, sticky="nw")
    phrase_input = tk.Text(main_frame, height=8, wrap="word")
    phrase_input.grid(row=3, column=0, pady=(6, 12), sticky="nsew")
    main_frame.rowconfigure(3, weight=1)

    prediction_label = ttk.Label(
        main_frame,
        text="Top predictions will appear here.",
        font=("TkDefaultFont", 12),
    )
    prediction_label.grid(row=4, column=0, pady=(0, 12), sticky="w")

    submit_button = ttk.Button(main_frame, text="Submit", state="disabled")
    submit_button.grid(row=5, column=0, sticky="e")

    training_results = Queue()
    model_holder = [None]

    def train_model():
        try:
            features, labels = load_data()
            trained_model = create_model1()
            trained_model.fit(features, labels)
            training_results.put(("success", trained_model))
        except Exception as error:
            training_results.put(("error", error))

    def check_training_status():
        try:
            result, value = training_results.get_nowait()
        except Empty:
            root.after(100, check_training_status)
            return

        if result == "error":
            status_label.config(text="The model could not be trained.")
            messagebox.showerror("Training error", str(value), parent=root)
            return

        model_holder[0] = value
        status_label.config(text="Ready to identify a phrase.")
        submit_button.config(state="normal")
        phrase_input.focus_set()

    def submit_phrase():
        phrase = phrase_input.get("1.0", "end").strip()
        if not phrase:
            messagebox.showwarning(
                "Missing phrase",
                "Enter a phrase before submitting.",
                parent=root,
            )
            return

        model = model_holder[0]
        if model is None:
            messagebox.showerror(
                "Prediction error",
                "The prediction model is not ready.",
                parent=root,
            )
            return

        try:
            predictions = get_top_predictions(
                model,
                phrase.replace("\n", " "),
            )
        except Exception as error:
            messagebox.showerror("Prediction error", str(error), parent=root)
            return

        prediction_label.config(
            text=f"Top predicted authors:\n{format_predictions(predictions)}"
        )

    submit_button.config(command=submit_phrase)
    Thread(target=train_model, daemon=True).start()
    root.after(100, check_training_status)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
