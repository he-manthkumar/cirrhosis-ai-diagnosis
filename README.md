```markdown
# AI-Powered Diagnosis of Cirrhosis Through Medical Data Analytics and Machine Learning

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Build Status](https://github.com/he-manthkumar/cirrhosis-ai-diagnosis/actions/workflows/ci.yml/badge.svg)](https://github.com/he-manthkumar/cirrhosis-ai-diagnosis/actions)

A full-stack research project and web application that combines a stacked ensemble of tree-based models with an LLM-powered explanation layer to provide interpretable, clinically meaningful predictions for cirrhosis stage from routine clinical data. This repository contains code, models, documentation, and deployment artifacts for a final-year research project aimed for academic publication.

Table of Contents
- Project Overview
- Novel Contribution
- Features
- Tech Stack
- Architecture
- Project Structure
- Installation
  - Prerequisites
  - Clone repository
  - Backend setup
  - Frontend setup
  - Database setup & migrations
  - Environment variables
  - Run locally
  - Docker deployment
- API Endpoints
- Dataset
- Model & Inference Pipeline
- Model Performance (placeholders)
- Experiments & Evaluation
- Screenshots (placeholders)
- Team
- Contributing
- License
- Acknowledgments
- Contact

---

Project Overview
----------------
- Purpose: Provide a clinician-friendly, research-grade application for predicting cirrhosis stage (1–4) from patient clinical features and for generating human-readable explanations to support clinical interpretation.
- Application type: Web application (React frontend + FastAPI backend) exposing REST endpoints for authentication, prediction, and history.
- Dataset: Kaggle "Cirrhosis Prediction" dataset (418 patients, ~20 clinical features).
- Research goal: Demonstrate an "interpretable-by-design" stacked ensemble that preserves predictive performance while producing LLM-crafted clinical narratives grounded in an interpretable decision-tree rule set.

Novel Contribution
------------------
- Interpretable-by-Design stacked ensemble architecture:
  - Layer 1 (base learners):
    - XGBoost
    - LightGBM
    - CatBoost
    - Shallow Decision Tree (max_depth=3) used specifically for interpretable rules
  - Layer 2 (meta learner): Logistic Regression trained on base learner predictions
- The Decision Tree extracts simple rule paths which are provided to an LLM (GPT-4 or an open-source LLM via HuggingFace) to generate human-readable clinical narratives that:
  - Explain why a particular stage was predicted
  - Highlight the most influential features
  - Offer cautions and confidence cues
- Fidelity concept: We compute a "tree-ensemble agreement" metric indicating how often the Decision Tree's rules align with the ensemble's final prediction — used as a fidelity badge to quantify explanation trustworthiness.
- Goal: Address the "black box" problem in medical AI while keeping state-of-the-art accuracy.

Key Features
------------
1. User Authentication (signup/login) using JWT (access + refresh tokens)
2. Patient Data Input Form capturing all 20 clinical features
3. Real-time Cirrhosis Stage Prediction (Stage 1–4)
4. Interactive Visualizations:
   - Probability bar chart for all stages
   - Decision Tree rule path display
   - LLM-generated clinical narrative explanation
   - Fidelity badge (tree-ensemble agreement metric)
5. Prediction History (persisted, paginated)
6. Case Study Viewer (for research paper and reproducibility)
7. Export predictions as JSON

Tech Stack
----------
Backend
- Python 3.10+
- FastAPI
- PostgreSQL 15+
- SQLAlchemy + Alembic
- scikit-learn, XGBoost, LightGBM, CatBoost
- OpenAI API or HuggingFace for LLM integration
- Async IO for LLM calls + caching layer (Redis or in-DB cache)

Frontend
- React 18+
- React Router
- Tailwind CSS
- Recharts (visualizations)
- Axios (API calls)

DevOps
- Docker & Docker Compose
- GitHub Actions (CI/CD)
- DBeaver (recommended for DB management)

Architecture
------------
- 3-tier system:
  - Frontend: React SPA (ports: 3000)
  - Backend: FastAPI (ports: 8000)
  - Persistence: PostgreSQL
- RESTful API:
  - Auth endpoints (signup/login/refresh)
  - Prediction endpoints (predict/history)
- Stacked ensemble inference pipeline:
  - Preprocessing -> Layer 1 predictions -> Meta-model -> Decision Tree rule extraction -> Async LLM explanation -> Response & cache
- Asynchronous LLM calls with result caching to avoid repeated API costs and reduce latency

Project Structure
-----------------
cirrhosis-ai-diagnosis/
├── backend/
│   ├── models/            # ML models, model loading & saving utilities
│   ├── routers/           # API endpoints (auth, predict, history)
│   ├── auth/              # JWT utilities, password hashing
│   ├── services/          # LLM service, model inference service
│   ├── utils/             # preprocessing, feature engineering, metrics
│   └── main.py            # FastAPI app
├── frontend/
│   ├── src/
│   │   ├── components/    # Reusable UI components
│   │   ├── pages/         # App pages (Login, Dashboard, Case Viewer)
│   │   └── services/      # API client (Axios), auth service
├── data/
│   ├── raw/               # Original Kaggle CSV (NOT included)
│   └── processed/         # Cleaned & feature-engineered CSVs
├── models/                # Trained model artifacts (pickle / joblib)
├── docs/                  # Research notes, evaluation reports, paper draft
├── docker-compose.yml
├── .env.example
└── README.md

Installation
------------
Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL 15+
- Docker & Docker Compose (optional but recommended for quick setup)
- (Optional) Redis for caching
- DBeaver (optional GUI for DB)

1) Clone repository
```bash
git clone https://github.com/he-manthkumar/cirrhosis-ai-diagnosis.git
cd cirrhosis-ai-diagnosis
```

2) Backend setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

3) Frontend setup
```bash
cd frontend
npm install
```

4) Database setup
- Create PostgreSQL database (example)
```sql
CREATE DATABASE cirrhosis_db;
CREATE USER cirrhosis_user WITH PASSWORD 'yourpassword';
GRANT ALL PRIVILEGES ON DATABASE cirrhosis_db TO cirrhosis_user;
```
- Run Alembic migrations
```bash
cd backend
alembic upgrade head
```

5) Environment variables
Copy .env.example to .env and fill values:
- DATABASE_URL=postgresql+asyncpg://cirrhosis_user:yourpassword@localhost:5432/cirrhosis_db
- SECRET_KEY=change_this_to_a_strong_secret
- ACCESS_TOKEN_EXPIRE_MINUTES=30
- REFRESH_TOKEN_EXPIRE_DAYS=7
- OPENAI_API_KEY=sk-...
- HUGGINGFACE_API_KEY=hf_...
- REDIS_URL=redis://localhost:6379
- FRONTEND_URL=http://localhost:3000

6) Run locally (development)
Start backend:
```bash
# from backend/
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Start frontend:
```bash
# from frontend/
npm run dev
# or
npm start
```
Open:
- Frontend: http://localhost:3000
- Backend docs: http://localhost:8000/docs

7) Docker deployment (recommended)
- Build and start services via docker-compose:
```bash
docker-compose up --build
```
Services typically exposed:
- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- PostgreSQL & Redis: internal

API Endpoints
-------------
- POST /api/auth/signup
  - Request: { email, password, name }
  - Response: user metadata
- POST /api/auth/login
  - Request: { email, password }
  - Response: { access_token, refresh_token, token_type }
- POST /api/auth/refresh
  - Request: { refresh_token }
  - Response: { access_token }
- POST /api/predict (protected)
  - Request: { patient_features: { Age, Sex, Albumin, ... } }
  - Response:
    - predicted_stage: 1|2|3|4
    - probabilities: {1:0.1,2:0.2,3:0.6,4:0.1}
    - decision_tree_path: ["If Albumin <= 3.2", "AND Bilirubin > 1.2", ...]
    - llm_explanation: "Clinical narrative..."
    - fidelity: 0.85
- GET /api/history (protected)
  - Response: list of previous predictions
- GET /api/prediction/{id} (protected)
  - Response: detailed record and exported JSON

Dataset
-------
- Source: Kaggle - Cirrhosis Prediction Dataset
  - URL: (add your local reference or Kaggle link in docs)
- Samples: 418 patients
- Typical Features (20; examples):
  - Age, Sex, Albumin, Bilirubin, Copper, Prothrombin, Stage, Status, Drug, Ascites, Hepatomegaly, Spiders, Class, Platelet, SGOT, SGPT, etc.
- Target:
  - Primary: Stage (1–4)
  - Secondary: Status (C, CL, D)
- Note: Raw dataset is expected to be placed under data/raw/ (not included in repo due to licensing)

Model & Inference Pipeline
--------------------------
- Preprocessing:
  - Imputation, scaling, categorical encoding, and domain-specific feature engineering
- Training:
  - Base learners: XGBoost, LightGBM, CatBoost, Shallow Decision Tree (max_depth=3)
  - Meta learner: Logistic Regression trained on the out-of-fold predictions of base learners (stacking)
  - Training scripts placed under backend/models/train_*.py
- Inference:
  - Input -> preprocessing -> base model predictions -> meta-model -> final prediction
  - Decision Tree path extraction (interpretable rule path) appended to response
  - Asynchronous LLM service consumes the extracted rule path and patient features to produce a narrative
  - Caching: identical (features + model hash) requests return cached LLM responses where available
- Explanation fidelity:
  - "Fidelity" metric = fraction of cases where Decision Tree's class (interpretable model) matches the final stacked ensemble prediction within top-k or exact match; stored with prediction

Model Performance (placeholders)
-------------------------------
These are placeholders — populate after experiments and model runs.

- Accuracy: 0.XX
- Macro F1-score: 0.XX
- AUC-ROC: 0.XX (per-class)
- Tree-Ensemble Agreement (Fidelity): 0.XX

Include in your paper and reports:
- Confusion matrices per stage
- Per-class precision/recall
- Calibration plots
- Decision curve analysis (optional for clinical utility)

Experiments & Evaluation
------------------------
- Cross-validation strategy: stratified k-fold (k=5 or 10)
- Baselines: single-model XGBoost, logistic regression, and shallow tree
- Ablation: measure effect of removing Decision Tree, or replacing LLM with template-based explanations
- Robustness: test on hold-out split and bootstrapped resamples

Screenshots (placeholders)
-------------------------
Below are placeholder images to be replaced with real screenshots in /docs/screenshots/

- Login page
  - [![Login Page](docs/screenshots/login-placeholder.png)](docs/screenshots/login-placeholder.png)
- Dashboard
  - [![Dashboard](docs/screenshots/dashboard-placeholder.png)](docs/screenshots/dashboard-placeholder.png)
- Prediction results
  - [![Prediction Results](docs/screenshots/prediction-placeholder.png)](docs/screenshots/prediction-placeholder.png)
- LLM explanation
  - [![LLM Explanation](docs/screenshots/explanation-placeholder.png)](docs/screenshots/explanation-placeholder.png)

Team
----
A three-person final year project team:
- Member A — Machine Learning & Backend
- Member B — Frontend & UX
- Member C — DevOps & Integration

Contributing
------------
Contributions are welcome. Please follow these guidelines:
1. Open an issue to discuss major changes before implementing them.
2. Create feature branches from main: git checkout -b feat/your-feature
3. Run tests and linting locally.
4. Open a pull request with clear description and linked issue.

Suggested development workflow:
- Use feature branches and PRs
- Keep model artifacts out of source control (store in /models or object storage)
- Use a .env file and never commit secrets

License
-------
This project is released under the MIT License. See LICENSE for details.

Acknowledgments
---------------
- Kaggle for the dataset
- FastAPI and React communities for excellent frameworks
- OpenAI / HuggingFace for LLM APIs and tools
- Mentors and reviewers for guidance during the research project

Contact
-------
For questions, issues, or collaboration:
- Repository: https://github.com/he-manthkumar/cirrhosis-ai-diagnosis
- Project lead: (add contact email or GitHub handle)

Notes for Reproducibility & Ethics
---------------------------------
- Keep patient data de-identified. Do not commit raw patient data into the repository.
- Include the dataset license and any IRB/ethics approvals in docs/ if applicable.
- This tool is research software. Clinical deployment requires extensive validation and regulatory approval.

Appendix: Example .env (backend/.env.example)
---------------------------------------------
```text
# Database
DATABASE_URL=postgresql+asyncpg://cirrhosis_user:yourpassword@localhost:5432/cirrhosis_db

# JWT / Security
SECRET_KEY=replace_with_a_strong_random_key
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# LLM
OPENAI_API_KEY=
HUGGINGFACE_API_KEY=
LLM_PROVIDER=openai  # or "huggingface"

# Caching
REDIS_URL=redis://localhost:6379

# App
FRONTEND_URL=http://localhost:3000
```

Thank you for reviewing this project README. For the next steps, consider:
- Running baseline training scripts to populate the models/ directory
- Filling the Model Performance section after evaluation
- Adding final screenshots and the research paper PDF under docs/

```
