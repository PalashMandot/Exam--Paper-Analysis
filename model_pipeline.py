"""Connect K-means, KNN, genuine ID3 and Random Forest to the website.

app.py -> backend.py -> this file -> the four model files.
Training labels always come from a later exam, never from the feature paper.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import io
import json
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from backend import csv_bytes, validate_questions
from id3_model import ID3Classifier
from kmeans_model import KMeansTopicModel
from knn_model import KNNRecurrenceModel
from random_forest_model import RandomForestRecurrenceModel

NUMERIC_FEATURES = ["appearance_count", "appearance_frequency", "gap_since_last",
                    "last_three_appearances", "topic_paper_count", "topic_paper_frequency", "average_marks"]


def fit_topics(frame):
    return KMeansTopicModel().fit(frame["question_text"].tolist(), frame["topic"].tolist())


def suggest_topics(frame):
    """Fill unassigned topics; leave question group identities and reviewed labels intact."""
    frame = validate_questions(frame)
    model = fit_topics(frame)
    proposed = model.predict_topics(frame["question_text"].tolist())
    labels = {}
    for record, suggestion in zip(frame.to_dict("records"), proposed):
        group = record["question_group_id"]
        if group not in labels:
            labels[group] = record["topic"] if record["topic"] != "Unassigned" else suggestion
        if record["topic"] != "Unassigned":
            labels[group] = record["topic"]
    frame["topic"] = frame["question_group_id"].map(labels)
    return frame


def feature_rows(frame, topic_model):
    """One row per known question group using only the supplied past papers."""
    orders = sorted(frame["exam_order"].unique().tolist())
    records = frame.to_dict("records")
    clusters = topic_model.predict_clusters(frame["question_text"].tolist())
    members = defaultdict(list)
    for record, cluster in zip(records, clusters):
        members[record["question_group_id"]].append((record, cluster))
    group_clusters = {group: Counter(c for _, c in rows).most_common(1)[0][0] for group, rows in members.items()}
    topic_papers = defaultdict(set)
    for group, rows in members.items():
        topic_papers[group_clusters[group]].update(row["exam_order"] for row, _ in rows)
    output = []
    for group in sorted(members):
        rows = [row for row, _ in members[group]]
        appearances = {row["exam_order"] for row in rows}
        known_topics = [row["topic"] for row in rows if row["topic"] != "Unassigned"]
        topic = known_topics[0] if known_topics else topic_model.topic_labels_.get(group_clusters[group], "Unassigned")
        marks = [float(row["marks"]) for row in rows if pd.notna(row["marks"])]
        topic_count = len(topic_papers[group_clusters[group]])
        output.append({
            "question_group_id": group, "question_text": rows[0]["question_text"], "topic": topic,
            "appearance_count": len(appearances), "appearance_frequency": len(appearances) / len(orders),
            "gap_since_last": len(orders) - 1 - max(orders.index(order) for order in appearances),
            "last_three_appearances": sum(order in appearances for order in orders[-3:]),
            "topic_paper_count": topic_count, "topic_paper_frequency": topic_count / len(orders),
            "average_marks": sum(marks) / len(marks) if marks else 0.0,
            "source_cutoff_order": int(orders[-1]),
        })
    return output


def numeric(row):
    return {name: float(row[name]) for name in NUMERIC_FEATURES}


def categorical(row):
    """Fixed categories match the genuine ID3 model in your supplied notebook."""
    count, gap = int(row["appearance_count"]), int(row["gap_since_last"])
    frequency, marks = row["topic_paper_frequency"], row["average_marks"]
    return {
        "appearance_count": str(count) if count <= 3 else "4+",
        "gap_since_last": str(gap) if gap <= 2 else "3-5" if gap <= 5 else "6+",
        "topic_frequency": "low" if frequency < 1 / 3 else "medium" if frequency < 2 / 3 else "high",
        "average_marks": "unknown" if marks == 0 else "low" if marks <= 5 else "medium" if marks <= 8 else "high",
    }


def historical_examples(frame, progress=None):
    orders = sorted(frame["exam_order"].unique().tolist())
    examples, splits = [], []
    for index in range(1, len(orders)):
        target_order = orders[index]
        past = frame[frame["exam_order"] < target_order]
        target = frame[frame["exam_order"] == target_order]
        if progress:
            progress(f"Building examples from papers before exam {target_order}…")
        # Vocabulary, TF-IDF and K-means see no text from the target or future exams.
        topics = fit_topics(past)
        rows = feature_rows(past, topics)
        next_groups = set(target["question_group_id"])
        known_groups = {row["question_group_id"] for row in rows}
        split = "train" if index < len(orders) - 2 else "validation" if index == len(orders) - 2 else "test"
        for row in rows:
            examples.append({**row, "target_order": int(target_order), "split": split,
                             "repeated_next": int(row["question_group_id"] in next_groups)})
        splits.append({"year": int(target.iloc[0]["year"]), "session": target.iloc[0]["session"],
                       "split": split, "history_papers": index, "examples": len(rows),
                       "new_question_groups": len(next_groups - known_groups)})
        if len(examples) > 10000:
            raise ValueError("This history creates more than 10,000 training rows. Use fewer papers or reviewed question groups for this student prototype.")
    return examples, splits


def evaluate(name, rows, scores, split):
    actual = [row["repeated_next"] for row in rows]
    predicted = [int(score >= 0.5) for score in scores]
    calibration = []
    for bucket in range(5):
        selected = [index for index, score in enumerate(scores) if min(4, int(score * 5)) == bucket]
        if selected:
            calibration.append({"predicted": float(np.mean([scores[i] for i in selected])),
                                "observed": float(np.mean([actual[i] for i in selected])), "count": len(selected)})
    return {"model": name, "split": split, "examples": len(rows),
            "precision": float(precision_score(actual, predicted, zero_division=0)),
            "recall": float(recall_score(actual, predicted, zero_division=0)),
            "f1": float(f1_score(actual, predicted, zero_division=0)),
            "brier": float(np.mean([(score - label) ** 2 for score, label in zip(scores, actual)])),
            "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
            "calibration": calibration}


def train_classifiers(examples):
    training = [row for row in examples if row["split"] == "train"]
    labels = [row["repeated_next"] for row in training]
    if set(labels) != {0, 1}:
        raise ValueError("The earlier training papers need both repeated and absent question groups. Add more papers. Frequency and topics are still available.")
    k = min(5, len(training))
    models = {"KNN": KNNRecurrenceModel(k), "ID3": ID3Classifier(max_depth=6),
              "Random Forest": RandomForestRecurrenceModel(n_estimators=200)}
    metrics = []
    for name, model in models.items():
        transform = categorical if name == "ID3" else numeric
        model.fit([transform(row) for row in training], labels)
        for split in ["validation", "test"]:
            rows = [row for row in examples if row["split"] == split]
            scores = model.predict_probabilities([transform(row) for row in rows])
            metrics.append(evaluate(name, rows, scores, split))
    # The newest paper is a final test. It never chooses the displayed model.
    validation = [metric for metric in metrics if metric["split"] == "validation"]
    selected = max(validation, key=lambda m: (m["f1"], m["precision"], -m["brier"]))["model"]
    # After evaluation, refit using all labelled past history for the next exam.
    for name, model in models.items():
        transform = categorical if name == "ID3" else numeric
        model.fit([transform(row) for row in examples], [row["repeated_next"] for row in examples])
    return models, metrics, selected


def analyze(frame, is_demo=False, progress=None):
    """One website action: topics and counts, then recurrence if history is sufficient."""
    frame = validate_questions(frame)
    if progress:
        progress("Grouping vocabulary with TF-IDF and K-means…")
    topic_model = fit_topics(frame)
    rows = feature_rows(frame, topic_model)
    groups = pd.DataFrame(rows)
    assigned = frame.copy()
    assigned["topic"] = assigned["question_group_id"].map(groups.set_index("question_group_id")["topic"])
    counts = assigned.groupby("topic")["exam_order"].nunique().sort_values(ascending=False)
    paper_count = frame["exam_order"].nunique()
    result = {"groups": groups, "topics": counts.rename("papers").reset_index(), "paper_count": int(paper_count),
              "kmeans": topic_model, "models": {}, "metrics": [], "examples": [], "splits": [],
              "selected_model": None, "is_demo": bool(is_demo), "reason": ""}
    if paper_count < 5:
        result["reason"] = "Topics and historical frequency are ready. Add at least five comparable papers to train, validate and test recurrence models."
        return result
    try:
        examples, splits = historical_examples(frame, progress)
        if progress:
            progress("Training KNN, ID3 and Random Forest, then comparing validation results…")
        models, metrics, selected = train_classifiers(examples)
    except ValueError as exc:
        result["reason"] = str(exc)
        return result
    for name, model in models.items():
        transform = categorical if name == "ID3" else numeric
        groups[name] = model.predict_probabilities([transform(row) for row in rows])
    groups["provisional_score"] = groups[selected]
    result.update(models=models, metrics=metrics, examples=examples, splits=splits, selected_model=selected)
    return result


def export_models(result):
    """Optional coursework download; every saved model is a separate flat file."""
    if not result["models"]:
        raise ValueError("Analyze a sufficient reviewed history before downloading trained models.")
    def dumped(value):
        buffer = io.BytesIO()
        joblib.dump(value, buffer, compress=3)
        return buffer.getvalue()
    files = {
        "trained_kmeans.joblib": dumped(result["kmeans"]),
        "trained_random_forest.joblib": dumped(result["models"]["Random Forest"]),
        "trained_knn.json": json.dumps(result["models"]["KNN"].to_dict(), indent=2, allow_nan=False).encode(),
        "trained_id3.json": json.dumps(result["models"]["ID3"].to_dict(), indent=2, allow_nan=False).encode(),
        "training_report.json": json.dumps({"selected_model": result["selected_model"], "is_demo": result["is_demo"],
                                            "score_kind": "provisional", "calibration_validated": False,
                                            "metrics": result["metrics"], "splits": result["splits"]}, indent=2, allow_nan=False).encode(),
        "historical_training_rows.csv": csv_bytes(pd.DataFrame(result["examples"])),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()
