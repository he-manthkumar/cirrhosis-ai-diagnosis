## Cirrhosis AI Diagnosis

Interpretable AI system for liver cirrhosis prognosis using a stacked ensemble model, rule-based explanations, optional image-assisted narrative generation, and JWT-based authentication.

### What This Project Includes

- FastAPI backend for auth, model training, and prediction APIs
- Stacked ensemble ML pipeline (XGBoost, LightGBM, CatBoost, Decision Tree + Logistic Regression)
- Explainability via decision-tree rule paths and key feature summaries
- Optional OpenAI narrative and image-assisted clinical explanation
- Frontend with:
	- Dedicated login page
	- Dedicated signup page
	- Main prediction dashboard
	- Profile page with prediction history search
	- Light/dark theme toggle
- Two SQLite databases by default:
	- Clinical data DB (`cirrhosis.db`)
	- Auth DB (`auth.db`)

### Tech Stack

- Backend: FastAPI, SQLAlchemy, Pydantic
- ML: scikit-learn, XGBoost, LightGBM, CatBoost
- Auth: JWT (`python-jose`), password hashing (`passlib`)
- Frontend: HTML/CSS/Vanilla JS
- DB: SQLite (default)

## Quick Start (Windows)

### 1) Clone and open project

```powershell
cd d:\Finalyearproject
git clone <your-repo-url> cirrhosis-ai-diagnosis
cd cirrhosis-ai-diagnosis
```

### 2) Create `.env`

Copy `.env.example` to `.env` and set values.

Minimum required variables:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

DATABASE_URL=sqlite:///./cirrhosis.db
AUTH_DATABASE_URL=sqlite:///./auth.db

JWT_SECRET_KEY=replace-with-a-strong-random-secret
JWT_EXPIRE_MINUTES=60

API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
```

### 3) Create virtual environment (Python 3.12 recommended)

```powershell
py -3.12 -m venv .venv
```

### 4) Install dependencies

```powershell
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### 5) Start backend

```powershell
.\.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Backend URLs:

- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

### 6) Start frontend (new terminal)

```powershell
cd frontend
python -m http.server 5500
```

Frontend URLs:

- Login: http://127.0.0.1:5500/login.html
- Signup: http://127.0.0.1:5500/signup.html
- Dashboard: http://127.0.0.1:5500/index.html
- Profile: http://127.0.0.1:5500/profile/

## Authentication Flow

1. Create account on `signup.html`
2. Login on `login.html`
3. JWT token is stored in browser local storage
4. Protected endpoints under `/predict/*` require Bearer token
5. Profile page uses token to fetch patient history

## First Run Notes

- If models are not available, prediction endpoints will return model-not-trained errors.
- Train model once using:

```powershell
cd d:\Finalyearproject\cirrhosis-ai-diagnosis
.\.venv\Scripts\python -m scripts.train_model
```

## API Overview

### Public/Auth

- `POST /auth/signup`
- `POST /auth/login`

### Training

- `POST /train/`
- `POST /train/sync`
- `GET /train/status`
- `GET /train/metrics`

### Prediction (JWT required)

- `POST /predict/`
- `POST /predict/explain`
- `POST /predict/full`
- `POST /predict/analyze-image`
- `GET /predict/feature-importance`
- `GET /predict/history/{patient_name}`

## Project Structure

```text
backend/
	auth_database.py      # Auth DB session/engine
	database.py           # Clinical DB session/engine
	routers/
		auth.py             # Signup/login endpoints
		prediction.py       # Protected prediction endpoints
		training.py         # Training endpoints
	services/
		auth_service.py     # JWT + password hashing
		ml_service.py       # Stacked ensemble train/infer
		explanation_service.py
		llm_service.py
	models/
		auth_models.py      # User table (auth DB)
		db_models.py        # Patient record table (clinical DB)
		patient.py          # Pydantic schemas

frontend/
	login.html
	signup.html
	index.html
	profile/index.html
	app.js
	auth.js
	theme.js
	styles.css
```

## Troubleshooting

### `pip install` fails on NumPy/scikit stack (Windows)

Use Python 3.12 environment:

```powershell
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### Auth hashing errors (`bcrypt` / 500 on signup)

Current implementation uses `pbkdf2_sha256` in auth service. Restart backend after pulling latest code.

### SQLite file not found

SQLite DB files are auto-created on backend startup:

- `cirrhosis.db`
- `auth.db`

## Security Notes

- Do not commit real `.env` secrets.
- Rotate API keys if exposed.
- Replace `JWT_SECRET_KEY` with a strong random value in production.

## License

See `LICENSE`.

