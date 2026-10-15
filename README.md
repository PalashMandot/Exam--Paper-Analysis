# Simple exam paper website

This is the complete Streamlit project. Every file is in this one folder. Keep the
files together; no frontend, backend or model subfolders are needed.

## 1. Run it on your computer

1. Install Python 3.12 if Python is not already installed. On Windows, select
   **Add Python to PATH** during installation.
2. Extract the ZIP and open the extracted folder in VS Code. You should see
   `app.py` and `requirements.txt` directly in the Explorer.
3. Select **Terminal > New Terminal**. The terminal must be inside this folder.
4. Install the packages once:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Start the website:

   ```bash
   python -m streamlit run app.py
   ```

6. Open the Local URL printed in the terminal, usually http://localhost:8501.

Leave the terminal open while using the website. Press **Ctrl+C** to stop it.
Next time, run only the start command. If Windows recognizes `py` instead of
`python`, replace `python` with `py` in both commands.

## 2. Try it first

1. Click **Use example data** in the sidebar.
2. Open **Analysis** and click **Analyze papers**.
3. See topic counts, historical frequencies, and provisional model scores.
4. Expand **Model comparison and training details** to see the three classifier
   results, confusion matrix, calibration check and training split.
5. Open **CSV to PDF**, choose **Use saved questions**, and click **Create PDF**.

The example CSV contains six hand-written sample papers, 36 question occurrences
and 12 question groups. It demonstrates the workflow. It is not real exam data.

## 3. Use your own CSV or PDF

1. Enter your subject, university/board and exam type in the sidebar.
2. In **Upload & review**, select your CSV or PDF files. Multiple files are allowed.
3. Set the year, session and order for PDFs or for CSVs without these columns.
4. For a CSV, select the question column. Recognized optional columns are used
   automatically. Use **Change other column mappings** if your headings differ.
5. Click **Read uploaded files**.
6. Correct the review table. Add missing questions, remove unwanted rows, and
   check marks and paper order. **Suggest topics with K-means** fills unassigned
   topics; you can rename its suggestions.
7. Check the review checkbox and click **Save reviewed questions**.
8. Open **Analysis** and click **Analyze papers**.

Use one subject, a consistent syllabus and comparable exam types in one session.
Your own uploads replace example data. Later uploads are added to your own saved
papers. Use **Clear all papers** when starting another subject.

**CSV format**

The included `example_history.csv` is a working template. One row is one question
occurrence in one paper. Questions appearing in several years need several rows.

| Column | What to enter |
| --- | --- |
| `question_text` | The full question. Required, or select the equivalent column. |
| `year` | Exam year, such as 2024. Set it in the form if absent. |
| `session` | Annual, Semester 1, Supplementary, etc. Defaults to Annual. |
| `exam_order` | A unique chronological number for the paper. Every question in that paper uses the same number. |
| `marks` | Optional marks, such as 5 or 8. Leave empty when unknown. |
| `topic` | Optional reviewed topic name. Blank topics can be suggested with K-means. |
| `question_group_id` | Optional ID for questions asking the same task. Blank IDs are generated from exact wording. |

For one paper per year, the default order is year × 100 + 1, such as 202401.
For two papers in 2024, use different sessions and orders such as 202401 and
202402. Older papers must have smaller order numbers.

Example: “Explain KNN” and “Describe how KNN classifies a sample” can share a
group ID after review. “Calculate distances for this KNN dataset” should use a
different group ID, even though it has the same topic. K-means groups topics; it
does not decide that two questions are equivalent.

Quote questions containing commas inside double quotes in a CSV. Excel handles
this when saving as CSV. The importer also accepts UTF-8, common Windows CSVs,
UTF-16, and comma, semicolon, tab or pipe separators.

**PDF support**

Searchable/text PDFs can be uploaded directly. The review table lets you correct
question boundaries and marks. Diagrams are not converted into model text.
Image-only/scanned PDFs need OCR first, or a CSV transcription. OCR software or
an AI service is not included in this small project.

**CSV to PDF**

Upload a CSV or use saved questions. Choose **Question paper** for numbered
questions, or **Table** for a general CSV. Select the relevant columns and click
**Create PDF**, then **Download PDF**. No external agent or API key is needed.
Use question layout if you want to upload the resulting PDF for analysis.

The included font files keep Latin text and many mathematical symbols readable.
The converter is primarily intended for English question papers; complex scripts
may require a different font and text shaping support.

## 4. What each separate file does

| File | Purpose |
| --- | --- |
| `app.py` | The Streamlit frontend: upload, review, results and PDF conversion screens. |
| `style.css` | Small visual adjustments to the website. |
| `backend.py` | Read CSV/PDF files, validate questions, export reviewed CSVs and generate PDFs. |
| `model_pipeline.py` | Create past-only features, train the models, compare them and generate results. |
| `kmeans_model.py` | TF-IDF and K-means topic grouping. |
| `knn_model.py` | A custom KNN classifier with training-only numeric scaling. |
| `id3_model.py` | Genuine ID3 using entropy and information gain, adapted from your supplied notebook. |
| `random_forest_model.py` | Random Forest classifier, adapted from your supplied notebook. |
| `requirements.txt` | Python packages to install. |
| `render.yaml` | Render deployment configuration. |
| `example_history.csv` | Small, clearly labelled example history and CSV template. |
| `DejaVuSans.ttf`, `DejaVuSans-Bold.ttf`, `FONT_LICENSE.txt` | Fonts and their license for readable PDFs. Keep these files with the code. |
| `README.md` | These instructions. |

The frontend imports the backend and model pipeline. The model pipeline imports
all four model files. K-means supplies topic clusters and their past paper counts;
KNN, ID3 and Random Forest use recurrence features to produce separate scores.

Streamlit builds the interface from Python, so the frontend entry point is
`app.py`. CSS is separate in `style.css`.

Your supplied notebook algorithms are used as the ID3 and Random Forest model
classes. ID3 keeps categorical, multiway information-gain splits. Random Forest
uses 200 trees and one worker to suit a small deployment. The source files are
model implementations, not models trained on your real exam papers. Clicking
**Analyze papers** trains them on the reviewed history in the current session.
The optional trained-model download provides four separate artifacts in a flat
ZIP, plus a report and historical training rows. It is not needed to run the site.
The site does not accept uploaded pickle/joblib models.

## 5. Deploy on Render

1. Create a GitHub repository. Upload **all files from this folder directly into
   the repository root**, including the fonts. Upload the extracted files, not
   the ZIP itself. `app.py` must be beside `requirements.txt`.
2. In Render, choose **New > Web Service** and connect that repository.
3. Select **Python** as the language. Leave **Root Directory** empty if the files
   are in the repository root.
4. Use this **Build Command**:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Use this **Start Command**:

   ```bash
   python -m streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true --server.maxUploadSize 10 --server.fileWatcherType none --browser.gatherUsageStats false
   ```

6. Add the environment variable `PYTHON_VERSION` with value `3.12.8`.
7. Set the health check path to `/_stcore/health` and create the web service.
8. Open the Render URL when deployment completes.

`render.yaml` contains these settings if you prefer creating a Render Blueprint.
Render setup reference: https://render.com/docs/web-services

## 6. How the predictions are evaluated

At least five comparable papers are needed for separate earlier training,
penultimate-paper validation and newest-paper testing. This minimum enables the
workflow; it does not make exam probabilities reliable.

Each training row describes a known question group using only earlier papers:
past appearance count, historical frequency, gap since last appearance,
appearances in the last three papers, K-means topic paper count and frequency,
and average marks. Unknown marks use zero for the model. ID3 receives fixed
categorical bins for count, gap, topic frequency and marks.

The label comes from the next paper: 1 if the known group appears, 0 if it does
not. Future-only groups are not added as past candidates. TF-IDF and K-means are
fitted again using past papers at each cutoff, and KNN scaling uses its training
rows only. Reviewed group equivalence must be based on question meaning rather
than whether it repeated in a later paper.

KNN, ID3 and Random Forest are compared on validation F1, then precision, then
lower Brier score. Their newest-paper test results are reported before refitting
the models on all labelled history for the next exam. Test results do not choose
the displayed model. Precision, recall, F1, a confusion matrix, Brier score and a
calibration plot are available in the website.

If there are fewer papers or the earlier training labels contain only one class,
the site still shows topics and historical frequency. It does not fabricate
training labels or model scores. All displayed model scores remain provisional;
calibrated exam probabilities need substantially more historical validation.

Each topic/question group counts at most once per paper for appearance frequency.
Topic repetition and question-task repetition are separate.

Uploaded papers and trained models stay in the current Streamlit session.
Download your reviewed CSV and optional models before refreshing, closing the
page or a service restart. Re-upload the CSV and analyze again to restore work.
Users' uploads are not written to shared project files. Limits: 10 MB per file,
100 PDF pages, 3,000 question rows, 40 papers and 10,000 generated training rows.
