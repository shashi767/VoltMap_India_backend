# VoltMap India

VoltMap India is a state-of-the-art predictive analytics platform designed to solve geospatial infrastructure imbalances for the Electric Vehicle (EV) market in India. It unifies predictive time-series forecasting (Prophet, Ridge Regression) with interactive geospatial supply/demand gap analysis.

For complete, detailed documentation on the platform's features, graphs, metrics, and machine learning integration, please see the full [VoltMap Explanation Guide](./VoltMap_Explanation.md).

## Project Structure

- **`/backend`**: Python FastAPI backend serving pre-trained machine learning models (`.pkl`) and static data CSVs.
- **`/frontend`**: React + Vite + Tailwind CSS Single Page Application (SPA).
- **`extracted_notebook.py`**: The original Jupyter Notebook execution code used to process the Vahan datasets and train the ML models.
- **`VoltMap_Explanation.md`**: Comprehensive documentation of the platform.

## Getting Started

To run the full application locally, you must start both the backend server and the frontend development server.

### 1. Start the Backend
The backend runs on FastAPI and uses Uvicorn. Ensure you have the `venv` activated and the requirements installed.
```bash
cd backend
source venv/bin/activate  # (or venv\Scripts\activate on Windows)
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Start the Frontend
The frontend uses Node.js and Vite. Open a new terminal window.
```bash
cd frontend
npm install
npm run dev
```

The application will be available at `http://localhost:5173`.
