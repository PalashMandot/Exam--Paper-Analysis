"""Streamlit frontend. Run this file with: python -m streamlit run app.py"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import altair as alt
import pandas as pd
import streamlit as st

from backend import COLUMNS, csv_bytes, guess_mapping, make_pdf, map_csv, read_csv, read_pdf, validate_questions
from model_pipeline import analyze, categorical, export_models, numeric, suggest_topics

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Examdesk | Paper analysis", page_icon="📄", layout="wide")
st.markdown("<style>" + (ROOT / "style.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)


def initialize():
    defaults = {"questions": None, "draft": None, "result": None, "is_demo": False, "draft_is_demo": False,
                "editor_version": 0, "upload_version": 0, "subject": "", "university": "", "exam_type": "End semester"}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_example():
    frame, _ = read_csv((ROOT / "example_history.csv").read_bytes())
    questions = map_csv(frame, guess_mapping(frame))
    st.session_state.update(questions=questions, draft=questions.copy(), result=None,
                            is_demo=True, draft_is_demo=True, subject="Machine Learning",
                            university="Example University", exam_type="End semester", extraction_messages=[])
    st.session_state.editor_version += 1
    st.session_state.upload_version += 1


def clear_papers():
    st.session_state.update(questions=None, draft=None, result=None, is_demo=False, draft_is_demo=False, extraction_messages=[])
    st.session_state.editor_version += 1
    st.session_state.upload_version += 1


def column_picker(label, options, guessed, key, optional=False):
    choices = (["Not present"] if optional else []) + list(options)
    index = choices.index(guessed) if guessed in choices else 0
    chosen = st.selectbox(label, choices, index=index, key=key)
    return None if optional and chosen == "Not present" else chosen


def upload_page():
    st.subheader("Add previous papers")
    st.write("Use papers from one subject and the same syllabus. A CSV can contain questions from several years.")
    st.download_button("Download example CSV", (ROOT / "example_history.csv").read_bytes(),
                       "example_history.csv", "text/csv", key="example_download")
    files = st.file_uploader("Upload CSV or PDF files", type=["csv", "pdf"], accept_multiple_files=True,
                             key=f"uploads_{st.session_state.upload_version}")
    st.caption("Up to 10 MB per file. PDFs must contain selectable text; scanned PDFs need OCR first.")
    if files and st.session_state.is_demo:
        st.caption("Your uploads will replace the example data when you save them.")
    inputs, errors = [], False
    for index, file in enumerate(files or []):
        key = f"file_{st.session_state.upload_version}_{index}"
        with st.expander(file.name, expanded=len(files) == 1):
            try:
                raw = file.getvalue()
                frame, warnings = read_csv(raw) if file.name.lower().endswith(".csv") else (None, [])
                for warning in warnings:
                    st.warning(warning)
                match = re.search(r"(?:19|20)\d{2}", file.name)
                default_year = int(match.group()) if match else 2025
                if frame is not None:
                    year_column = guess_mapping(frame)["year"]
                    values = pd.to_numeric(frame[year_column], errors="coerce") if year_column else pd.Series(dtype=float)
                    if values.notna().all() and values.nunique() == 1:
                        value = float(values.iloc[0])
                        if value.is_integer() and 1950 <= value <= 2100:
                            default_year = int(value)
                first, second, third = st.columns(3)
                year = first.number_input("Year if missing", 1950, 2100, default_year, key=key + "_year")
                session = second.text_input("Session if missing", "Annual", key=key + "_session")
                order = third.number_input("Exam order if missing", 1, 9999999, default_year * 100 + 1,
                                            key=key + "_order", help="Older papers need smaller numbers. For two exams in 2024, use 202401 and 202402.")
                # Changing the year also changes the default order unless you supplied an order explicitly.
                if order == default_year * 100 + 1:
                    order = int(year) * 100 + 1
                mapping = None
                if frame is not None:
                    mapping = guess_mapping(frame)
                    mapping["question_text"] = column_picker("Question column", frame.columns, mapping["question_text"], key + "_question")
                    st.dataframe(frame.head(3), hide_index=True, width="stretch")
                    st.caption("Recognized year, session, marks, topic and group columns are used automatically.")
                    if st.checkbox("Change other column mappings", key=key + "_map"):
                        first, second = st.columns(2)
                        for number, field in enumerate([c for c in COLUMNS if c != "question_text"]):
                            with first if number % 2 == 0 else second:
                                mapping[field] = column_picker(field.replace("_", " ").title(), frame.columns,
                                                               mapping[field], key + "_" + field, optional=True)
                inputs.append((file.name, raw, frame, mapping, int(year), session, int(order)))
            except ValueError as exc:
                errors = True
                st.error(str(exc))
    if files and st.button("Read uploaded files", type="primary", disabled=errors):
        if not st.session_state.subject.strip():
            st.error("Enter the subject in the sidebar first.")
        else:
            try:
                imported = []
                messages = []
                for name, raw, frame, mapping, year, session, order in inputs:
                    if frame is not None:
                        questions = map_csv(frame, mapping, year, session, order)
                    else:
                        questions, warnings = read_pdf(raw, year, session, order)
                        messages.extend(f"{name}: {warning}" for warning in warnings)
                    imported.append(questions)
                current = st.session_state.questions
                if current is not None and not st.session_state.is_demo:
                    imported.insert(0, current)
                st.session_state.draft = validate_questions(pd.concat(imported, ignore_index=True))
                st.session_state.draft_is_demo = False
                st.session_state.editor_version += 1
                st.session_state["extraction_messages"] = messages
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    draft = st.session_state.draft
    if draft is None:
        st.info("Upload your files above, or choose Use example data in the sidebar.")
        return
    st.divider()
    st.subheader("Review the questions")
    for message in st.session_state.get("extraction_messages", []):
        st.warning(message)
    st.caption("Correct text, years, sessions and marks. You can add or delete rows. Blank group IDs are filled automatically.")
    st.caption("Give similar questions the same group ID only when they ask the same task. Questions on the same topic can have different group IDs.")
    edited = st.data_editor(draft, num_rows="dynamic", hide_index=True, width="stretch", height=370,
                            key=f"review_{st.session_state.editor_version}", column_order=COLUMNS,
                            column_config={
                                "question_text": st.column_config.TextColumn("Question", width="large"),
                                "question_group_id": st.column_config.TextColumn("Question group ID", width="medium"),
                                "marks": st.column_config.NumberColumn("Marks", min_value=0, max_value=1000),
                                "exam_order": st.column_config.NumberColumn("Exam order", format="%d"),
                                "year": st.column_config.NumberColumn("Year", format="%d"),
                            })
    left, right = st.columns(2)
    if left.button("Suggest topics with K-means", key="suggest_topics"):
        try:
            with st.spinner("Finding questions with similar vocabulary…"):
                st.session_state.draft = suggest_topics(edited)
            st.session_state.editor_version += 1
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    reviewed = st.checkbox("I checked the questions and these papers follow the same syllabus.",
                           key=f"reviewed_{st.session_state.editor_version}")
    if right.button("Save reviewed questions", type="primary", disabled=not reviewed, key="save_questions"):
        try:
            questions = validate_questions(edited)
            st.session_state.questions = questions
            st.session_state.draft = questions.copy()
            st.session_state.is_demo = st.session_state.draft_is_demo
            st.session_state.result = None
            st.session_state.editor_version += 1
            st.session_state["extraction_messages"] = []
            st.session_state["saved_notice"] = True
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if st.session_state.pop("saved_notice", False):
        st.success("Questions saved. Open Analysis and select Analyze papers.")
    if st.session_state.questions is not None:
        st.download_button("Download my reviewed CSV", csv_bytes(st.session_state.questions),
                           "reviewed_questions.csv", "text/csv", key="reviewed_download")


def model_details(result):
    with st.expander("Model comparison and training details"):
        st.write("K-means groups vocabulary. KNN, ID3 and Random Forest predict whether a known question group repeats in the next paper.")
        st.caption("Earlier target exams train the models, the penultimate exam selects the model, and the newest exam provides the final test.")
        split = st.selectbox("Show evaluation on", ["Final test paper", "Validation paper"], key="metric_split")
        split_key = "test" if split == "Final test paper" else "validation"
        metrics = [row for row in result["metrics"] if row["split"] == split_key]
        table = pd.DataFrame(metrics)[["model", "examples", "precision", "recall", "f1", "brier"]]
        st.dataframe(table, hide_index=True, width="stretch")
        model_name = st.selectbox("Inspect classifier", ["KNN", "ID3", "Random Forest"], key="inspect_model")
        metric = next(row for row in metrics if row["model"] == model_name)
        left, right = st.columns(2)
        with left:
            st.write("Confusion matrix")
            st.dataframe(pd.DataFrame(metric["confusion_matrix"], index=["Actually absent", "Actually repeated"],
                                     columns=["Predicted absent", "Predicted repeat"]), width="stretch")
            st.caption("Precision: among predicted repeats, how many actually appeared. Recall: among actual repeats, how many were found.")
        with right:
            st.write("Calibration check")
            points = pd.DataFrame(metric["calibration"])
            chart = alt.Chart(points).mark_circle(size=90, color="#365F50").encode(
                x=alt.X("predicted:Q", title="Mean model score", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("observed:Q", title="Actual recurrence rate", scale=alt.Scale(domain=[0, 1])),
                tooltip=["predicted:Q", "observed:Q", "count:Q"])
            diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="#929B94").encode(x="x:Q", y="y:Q")
            st.altair_chart(chart + diagonal, use_container_width=True)
            st.caption("Few examples make this check uncertain. Displayed scores remain provisional; they are not calibrated exam probabilities.")
        st.write("Chronological training split")
        st.dataframe(pd.DataFrame(result["splits"]), hide_index=True, width="stretch")
        st.write("K-means topic suggestions")
        cluster_rows = result["kmeans"].metadata()["clusters"]
        st.dataframe(pd.DataFrame([{**row, "keywords": ", ".join(row["keywords"])} for row in cluster_rows]),
                     hide_index=True, width="stretch")
        st.write("Explain a question's scores")
        groups = result["groups"].set_index("question_group_id")
        group_id = st.selectbox("Question group", groups.index.tolist(),
                                format_func=lambda group: groups.loc[group, "question_text"][:100], key="explain_group")
        row = groups.loc[group_id].to_dict()
        neighbors = result["models"]["KNN"].nearest_neighbors(numeric(row))
        rules = result["models"]["ID3"].decision_path(categorical(row))
        reasons = {
            "KNN": f"{sum(n['label'] for n in neighbors)} of {len(neighbors)} nearest standardized historical examples repeated in their next paper.",
            "ID3": "; ".join(rules) or "Use the training class distribution at a leaf.",
            "Random Forest": "Average class score from 200 trees trained on past recurrence features.",
        }
        st.dataframe(pd.DataFrame([{"Model": name, "Provisional score (%)": round(float(row[name]) * 100, 1),
                                    "Explanation": reason} for name, reason in reasons.items()]), hide_index=True, width="stretch")
        st.download_button("Download the four trained models", export_models(result),
                           "trained_models.zip", "application/zip", key="models_download")


def analysis_page():
    st.subheader("Paper analysis")
    frame = st.session_state.questions
    if frame is None:
        st.info("Add and save your questions first. You can also try the example data.")
        return
    st.caption("Analysis uses your saved questions. Save review-table edits before running it again.")
    if st.button("Analyze papers", type="primary", key="analyze_papers"):
        try:
            with st.spinner("Analyzing your reviewed papers…"):
                status = st.empty()
                st.session_state.result = analyze(frame, st.session_state.is_demo, progress=status.caption)
                status.empty()
        except ValueError as exc:
            st.error(str(exc))
    result = st.session_state.result
    if result is None:
        st.write("Select Analyze papers to see topic frequency and questions to revise.")
        return
    if result["is_demo"]:
        st.info("Example data: these hand-written sample papers demonstrate the project. The scores are not real exam predictions.")
    first, second, third = st.columns(3)
    first.metric("Papers", result["paper_count"])
    second.metric("Question groups", len(result["groups"]))
    third.metric("Topics", len(result["topics"]))
    st.subheader("Frequently tested topics")
    topic_table = result["topics"].copy()
    topic_table["Historical frequency (%)"] = (topic_table["papers"] / result["paper_count"] * 100).round(1)
    topic_table = topic_table.rename(columns={"topic": "Topic", "papers": "Papers containing this topic"})
    st.dataframe(topic_table, hide_index=True, width="stretch")
    st.caption("Each topic is counted once per paper, even if that paper has several questions from the topic.")
    if result["reason"]:
        st.info(result["reason"])
    elif result["selected_model"]:
        st.success(f"All four models ran. {result['selected_model']} supplies the displayed score, selected using validation results.")
        st.caption("A historical frequency describes the past. A provisional model score estimates recurrence in the next comparable exam; it is not a guarantee.")
    st.subheader("Questions to revise")
    left, right = st.columns(2)
    topic = left.selectbox("Topic", ["All topics"] + sorted(result["groups"]["topic"].unique()), key="topic_filter")
    search = right.text_input("Find a question", placeholder="Type a word or phrase", key="question_search")
    groups = result["groups"].copy()
    if topic != "All topics":
        groups = groups[groups["topic"] == topic]
    if search.strip():
        groups = groups[groups["question_text"].str.contains(search.strip(), case=False, regex=False)]
    sort_column = "provisional_score" if result["models"] else "appearance_frequency"
    groups = groups.sort_values(sort_column, ascending=False, kind="stable")
    display = groups[["topic", "question_text", "appearance_count", "appearance_frequency"]].copy()
    display["appearance_frequency"] = (display["appearance_frequency"] * 100).round(1)
    display.columns = ["Topic", "Question", "Papers containing this question", "Historical frequency (%)"]
    if result["models"]:
        display["Provisional model score (%)"] = (groups["provisional_score"] * 100).round(1)
    st.dataframe(display, hide_index=True, width="stretch", column_config={"Question": st.column_config.TextColumn(width="large")})
    st.download_button("Download these results", csv_bytes(display), "question_analysis.csv", "text/csv", key="results_download")
    if result["models"]:
        model_details(result)


def conversion_page():
    st.subheader("Convert CSV to PDF")
    st.write("Turn a question list or a general CSV table into a readable PDF. No API key is needed.")
    source = st.radio("Choose a source", ["Upload a CSV", "Use saved questions"], horizontal=True, key="converter_source")
    if source == "Upload a CSV":
        file = st.file_uploader("CSV to convert", type=["csv"], key="converter_file")
        if file is None:
            return
        try:
            frame, warnings = read_csv(file.getvalue())
        except ValueError as exc:
            st.error(str(exc))
            return
        for warning in warnings:
            st.warning(warning)
    else:
        frame = st.session_state.questions
        if frame is None:
            st.info("Save your questions in Upload & review first.")
            return
    st.dataframe(frame.head(5), hide_index=True, width="stretch")
    title = st.text_input("PDF title", value="Questions to revise", key="pdf_title", max_chars=120)
    layout = st.selectbox("PDF layout", ["Question paper", "Table"], key="pdf_layout")
    mapping = guess_mapping(frame)
    question_column, marks_column, columns = None, None, None
    if layout == "Question paper":
        question_column = column_picker("Questions column", frame.columns, mapping["question_text"], "pdf_questions")
        marks_column = column_picker("Marks column", frame.columns, mapping["marks"], "pdf_marks", optional=True)
        st.caption("Choose this layout if you plan to upload the generated PDF for analysis.")
    else:
        columns = st.multiselect("Columns to include", frame.columns.tolist(), default=frame.columns.tolist()[:6], key="pdf_columns")
    signature = hashlib.sha256(frame.to_csv(index=False).encode() + str((title, layout, question_column, marks_column, columns)).encode()).hexdigest()
    if st.button("Create PDF", type="primary", key="create_pdf"):
        try:
            with st.spinner("Creating your PDF…"):
                data = make_pdf(frame, title, "questions" if layout == "Question paper" else "table",
                                question_column, marks_column, columns)
            st.session_state["converted_pdf"] = (signature, data)
        except ValueError as exc:
            st.error(str(exc))
    converted = st.session_state.get("converted_pdf")
    if converted and converted[0] == signature:
        st.success("Your PDF is ready.")
        st.download_button("Download PDF", converted[1], "questions.pdf", "application/pdf", key="pdf_download")


def main():
    initialize()
    with st.sidebar:
        st.title("Examdesk")
        st.caption("Previous papers, clearer revision.")
        st.text_input("Subject", key="subject", placeholder="e.g. Machine Learning")
        st.text_input("University / board", key="university", placeholder="Your university")
        st.text_input("Exam type", key="exam_type")
        st.button("Use example data", on_click=load_example, width="stretch")
        st.button("Clear all papers", on_click=clear_papers, width="stretch")
        st.divider()
        if st.session_state.questions is not None:
            st.caption(f"{st.session_state.questions['exam_order'].nunique()} saved papers · {len(st.session_state.questions)} question rows")
        st.caption("Your work stays in this browser session. Download your reviewed CSV before refreshing or closing the page.")
    st.title("Previous-year paper analysis")
    st.write("Find repeated topics, review past questions, and compare recurrence models.")
    if st.session_state.is_demo:
        st.caption("EXAMPLE DATA · Hand-written Machine Learning questions")
    upload, analysis_tab, convert = st.tabs(["Upload & review", "Analysis", "CSV to PDF"])
    with upload:
        upload_page()
    with analysis_tab:
        analysis_page()
    with convert:
        conversion_page()


if __name__ == "__main__":
    main()
