

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold, GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_ITER_SEARCH = 10
DATASET_PATH = "splt_dataset.xlsx"
RESULTS_DIR = Path("lab09_results")




def load_project_dataset(excel_path):
    """Load and validate the writer-identification dataset."""
    df = pd.read_excel(excel_path)

    required = {"Author", "Blog"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["Author", "Blog"]).copy()
    df["Author"] = df["Author"].astype(str)
    df["Blog"] = df["Blog"].astype(str)
    return df


def inspect_dataset(df):
    
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "authors": df["Author"].nunique(),
        "distribution": df["Author"].value_counts()
    }


def split_project_dataset(df):
    
    return train_test_split(
        df["Blog"],
        df["Author"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["Author"]
    )


def create_tfidf_vectorizer(max_features=5000):
   
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True
    )


def transform_text_data(X_train, X_test):
   
    vectorizer = create_tfidf_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer




def create_perceptron_search():
   
    pipeline = Pipeline([
        ("tfidf", create_tfidf_vectorizer()),
        ("classifier", Perceptron(random_state=RANDOM_STATE))
    ])

    parameters = {
        "tfidf__max_features": [1000, 2000, 3000, 5000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "classifier__penalty": [None, "l2", "l1", "elasticnet"],
        "classifier__alpha": [0.00001, 0.0001, 0.001, 0.01],
        "classifier__max_iter": [500, 1000, 2000],
        "classifier__eta0": [0.1, 0.5, 1.0]
    }

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    return RandomizedSearchCV(
        pipeline,
        parameters,
        n_iter=N_ITER_SEARCH,
        scoring="f1_weighted",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True
    )


def create_mlp_search():
  
    pipeline = Pipeline([
        ("tfidf", create_tfidf_vectorizer()),
        ("classifier", MLPClassifier(
            random_state=RANDOM_STATE,
            early_stopping=False
        ))
    ])

    parameters = {
        "tfidf__max_features": [500, 1000, 2000, 3000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "classifier__hidden_layer_sizes": [
            (50,), (100,), (50, 25), (100, 50)
        ],
        "classifier__activation": ["relu", "tanh"],
        "classifier__learning_rate_init": [0.0001, 0.001, 0.01],
        "classifier__alpha": [0.0001, 0.001, 0.01],
        "classifier__max_iter": [200, 300, 500]
    }

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    return RandomizedSearchCV(
        pipeline,
        parameters,
        n_iter=N_ITER_SEARCH,
        scoring="f1_weighted",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True
    )


def run_random_search(search, X_train, y_train):
    """Fit a RandomizedSearchCV object."""
    search.fit(X_train, y_train)
    return search




def create_classifiers():
    """Create classifiers requested by Lab 09."""
    classifiers = {
        "SVM": LinearSVC(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE
        ),
        "Naive Bayes": MultinomialNB()
    }

    try:
        from catboost import CatBoostClassifier
        classifiers["CatBoost"] = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass",
            verbose=False,
            random_seed=RANDOM_STATE
        )
    except ImportError:
        pass

    try:
        from xgboost import XGBClassifier
        classifiers["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            objective="multi:softmax",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    except ImportError:
        pass

    return classifiers


def train_classifiers(classifiers, X_train, y_train):
    """Train all available classifiers."""
    trained = {}

    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


def calculate_metrics(y_true, y_pred):
   
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "F1 Score": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
    }


def evaluate_classifiers(models, X_test, y_test):
    
    rows = []

    for name, model in models.items():
        predictions = model.predict(X_test)
        metrics = calculate_metrics(y_test, predictions)
        metrics["Classifier"] = name
        rows.append(metrics)

    return pd.DataFrame(rows)[
        ["Classifier", "Accuracy", "Precision", "Recall", "F1 Score"]
    ].sort_values("F1 Score", ascending=False).reset_index(drop=True)


def get_classification_reports(models, X_test, y_test):
    """Generate detailed classification reports."""
    reports = {}

    for name, model in models.items():
        predictions = model.predict(X_test)
        reports[name] = classification_report(
            y_test, predictions, zero_division=0
        )

    return reports




def create_text_pipeline(classifier):
    
    return Pipeline([
        ("tfidf", create_tfidf_vectorizer()),
        ("classifier", classifier)
    ])


def train_pipeline(X_train, y_train, classifier):
    
    pipeline = create_text_pipeline(classifier)
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_pipeline(pipeline, X_test, y_test):
    
    predictions = pipeline.predict(X_test)
    return calculate_metrics(y_test, predictions)


def train_lime_model(X_train, y_train):
  
    model = Pipeline([
        ("tfidf", create_tfidf_vectorizer()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ])
    model.fit(X_train, y_train)
    return model


def create_lime_explanation(model, text, class_names, num_features=10):
    
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError as exc:
        raise ImportError(
            "LIME is not installed. Run: pip install lime"
        ) from exc

    explainer = LimeTextExplainer(
        class_names=list(class_names),
        random_state=RANDOM_STATE
    )

    return explainer.explain_instance(
        text,
        model.predict_proba,
        num_features=num_features
    )


def save_lime_explanation(explanation, output_path):
    """Save LIME explanation as an HTML file."""
    explanation.save_to_file(str(output_path))



def create_shap_explainer(model):
   
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Run: pip install shap"
        ) from exc

    return shap.Explainer(model.predict)


def create_grid_search():
   
    pipeline = Pipeline([
        ("tfidf", create_tfidf_vectorizer()),
        ("classifier", Perceptron(random_state=RANDOM_STATE))
    ])

    parameters = {
        "classifier__penalty": [None, "l2"],
        "classifier__alpha": [0.0001, 0.001],
        "classifier__max_iter": [500, 1000]
    }

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    return GridSearchCV(
        pipeline,
        parameters,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1
    )



def save_dataframe(dataframe, filename):
    
    RESULTS_DIR.mkdir(exist_ok=True)
    dataframe.to_csv(RESULTS_DIR / filename, index=False)


def save_search_results(search, filename):
   
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(search.cv_results_).to_csv(
        RESULTS_DIR / filename,
        index=False
    )



def main():


    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 75)
    print("23CSE301 - LAB 09")
    print("WRITER IDENTIFICATION FROM TEXT BLOGS")
    print("=" * 75)

 

    print("\n" + "=" * 75)
    print("A1. PROJECT DATASET")
    print("=" * 75)

    df = load_project_dataset(DATASET_PATH)
    information = inspect_dataset(df)

    print(f"Number of records : {information['rows']}")
    print(f"Number of authors : {information['authors']}")
    print(f"Columns           : {information['columns']}")

    print("\nAuthor distribution:")
    print(information["distribution"])

    X_train_text, X_test_text, y_train, y_test = split_project_dataset(df)

    print(f"\nTraining samples: {len(X_train_text)}")
    print(f"Testing samples : {len(X_test_text)}")



    print("\n" + "=" * 75)
    print("A2. RANDOMIZEDSEARCHCV - PERCEPTRON")
    print("=" * 75)

    perceptron_search = run_random_search(
        create_perceptron_search(),
        X_train_text,
        y_train
    )

    print("Best parameters:")
    print(perceptron_search.best_params_)
    print(
        f"Best CV weighted F1: "
        f"{perceptron_search.best_score_:.4f}"
    )

    perceptron_pred = perceptron_search.predict(X_test_text)
    perceptron_metrics = calculate_metrics(y_test, perceptron_pred)

    print("\nTest performance:")
    for metric, value in perceptron_metrics.items():
        print(f"{metric:10s}: {value:.4f}")

    save_search_results(
        perceptron_search,
        "perceptron_randomized_search.csv"
    )

  

    print("\n" + "=" * 75)
    print("A2. RANDOMIZEDSEARCHCV - MLP")
    print("=" * 75)

    mlp_search = run_random_search(
        create_mlp_search(),
        X_train_text,
        y_train
    )

    print("Best parameters:")
    print(mlp_search.best_params_)
    print(
        f"Best CV weighted F1: "
        f"{mlp_search.best_score_:.4f}"
    )

    mlp_pred = mlp_search.predict(X_test_text)
    mlp_metrics = calculate_metrics(y_test, mlp_pred)

    print("\nTest performance:")
    for metric, value in mlp_metrics.items():
        print(f"{metric:10s}: {value:.4f}")

    save_search_results(
        mlp_search,
        "mlp_randomized_search.csv"
    )



    print("\n" + "=" * 75)
    print("A3. CLASSIFIER COMPARISON")
    print("=" * 75)

    X_train_tfidf, X_test_tfidf, vectorizer = transform_text_data(
        X_train_text,
        X_test_text
    )

    classifiers = create_classifiers()

    print("Classifiers:")
    for name in classifiers:
        print(f"- {name}")

    trained_models = train_classifiers(
        classifiers,
        X_train_tfidf,
        y_train
    )

    comparison = evaluate_classifiers(
        trained_models,
        X_test_tfidf,
        y_test
    )

    print("\nPerformance comparison:")
    print(comparison.to_string(index=False))

    save_dataframe(
        comparison,
        "classifier_comparison.csv"
    )

    reports = get_classification_reports(
        trained_models,
        X_test_tfidf,
        y_test
    )

    for name, report in reports.items():
        print("\n" + "-" * 75)
        print(name)
        print("-" * 75)
        print(report)

    # Add tuned Perceptron and MLP to the comparison table.
    tuned_results = pd.DataFrame([
        {"Classifier": "Tuned Perceptron", **perceptron_metrics},
        {"Classifier": "Tuned MLP", **mlp_metrics}
    ])

    combined = pd.concat(
        [comparison, tuned_results],
        ignore_index=True
    ).sort_values(
        "F1 Score",
        ascending=False
    )

    print("\nCombined results:")
    print(combined.to_string(index=False))

    save_dataframe(
        combined,
        "combined_classifier_results.csv"
    )



    print("\n" + "=" * 75)
    print("A4. TEXT CLASSIFICATION PIPELINE")
    print("=" * 75)

    pipeline = train_pipeline(
        X_train_text,
        y_train,
        LinearSVC(random_state=RANDOM_STATE)
    )

    pipeline_metrics = evaluate_pipeline(
        pipeline,
        X_test_text,
        y_test
    )

    for metric, value in pipeline_metrics.items():
        print(f"{metric:10s}: {value:.4f}")

    pipeline_pred = pipeline.predict(X_test_text)

    print("\nPipeline classification report:")
    print(
        classification_report(
            y_test,
            pipeline_pred,
            zero_division=0
        )
    )



    print("\n" + "=" * 75)
    print("A5. LIME EXPLANATION")
    print("=" * 75)

    try:
        lime_model = train_lime_model(
            X_train_text,
            y_train
        )

        text = X_test_text.iloc[0]
        classes = sorted(y_train.unique())

        explanation = create_lime_explanation(
            lime_model,
            text,
            classes,
            num_features=10
        )

        output_file = RESULTS_DIR / "lime_explanation.html"
        save_lime_explanation(
            explanation,
            output_file
        )

        predicted_author = lime_model.predict([text])[0]

        print(f"Predicted writer: {predicted_author}")
        print(f"LIME file: {output_file}")

        print("\nImportant words/features:")
        for word, weight in explanation.as_list():
            print(f"{word:30s} {weight:+.6f}")

    except ImportError as error:
        print(error)

    print("\n" + "=" * 75)
    print("OPTIONAL O3. GRIDSEARCHCV")
    print("=" * 75)

    run_grid_search = False

    if run_grid_search:
        grid_search = create_grid_search()
        grid_search.fit(X_train_text, y_train)

        print("Best GridSearchCV parameters:")
        print(grid_search.best_params_)
        print(
            f"Best GridSearchCV weighted F1: "
            f"{grid_search.best_score_:.4f}"
        )

        save_search_results(
            grid_search,
            "perceptron_grid_search.csv"
        )
    else:
        print(
            "GridSearchCV is disabled by default. "
            "Set run_grid_search = True to execute O3."
        )

    print("\n" + "=" * 75)
    print("LAB 09 COMPLETED")
    print("=" * 75)
    print(f"Results folder: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
