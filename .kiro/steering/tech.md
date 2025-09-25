# Technology Stack & Build System

## Backend Stack
- **Framework**: FastAPI with Uvicorn ASGI server
- **Language**: Python 3.10+
- **Database**: SQLite with SQLAlchemy ORM
- **AI/ML**: PyTorch (CPU-only), Ultralytics YOLOv8, scikit-learn, Transformers
- **Computer Vision**: OpenCV, ONNX Runtime for optimization
- **Object Tracking**: ByteTrack, YOLOX, motmetrics
- **API Documentation**: OpenAPI/Swagger auto-generated

## Frontend Stack
- **Framework**: React 19+ with TypeScript
- **Build Tool**: Vite 7+ with HMR
- **Styling**: Tailwind CSS 3+
- **Routing**: React Router DOM 7+
- **HTTP Client**: Axios with retry logic and interceptors
- **UI Components**: Headless UI, Heroicons
- **Charts**: Recharts for data visualization

## Development Tools
- **Code Quality**: ESLint, Black (Python), isort
- **Testing**: pytest (backend), Jest/React Testing Library (frontend)
- **Containerization**: Docker with multi-stage builds
- **Process Management**: Docker Compose for orchestration

## Common Commands

### Backend Development
```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Start API server
python start_api.py
# or
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
python -m pytest tests/ -v

# Code formatting
black . && isort .
```

### Frontend Development
```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

### Docker Operations
```bash
# Build and run with compose
docker-compose up --build

# Build production image
docker build -t hifazat-ai:latest .

# Run with resource limits
docker run -d --name hifazat-ai --memory=6g --cpus=4 -p 8000:8000 hifazat-ai:latest
```

### Evaluation & Testing
```bash
# Setup evaluation environment
cd backend/evaluation
python setup_evaluation.py

# Run comprehensive evaluation
cd scripts
python run_evaluation.py

# Generate test data
python test_data_manager.py --action generate --type threat --samples 200
```

## CPU Optimization Settings
For i5 6th gen target hardware:
```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

## Environment Variables
- `VITE_API_URL`: Frontend API endpoint (default: http://localhost:8000)
- `DATABASE_URL`: Database connection string
- `LOG_LEVEL`: Logging verbosity (INFO, DEBUG, ERROR)
- `CORS_ORIGINS`: Allowed frontend origins

## Project Structure
```
hifazat-ai/
├── backend/                    # Python FastAPI backend
│   ├── api/                   # API routes and models
│   ├── pipelines/             # AI processing pipelines
│   │   ├── threat/           # Threat intelligence pipeline
│   │   ├── video/            # Video surveillance pipeline
│   │   └── anomaly/          # Border anomaly detection
│   ├── utils/                # Shared utilities
│   ├── evaluation/           # Model evaluation framework
│   └── demo/                 # Demo scripts and data
├── frontend/                  # React TypeScript frontend
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── services/         # API and utility services
│   │   ├── types/           # TypeScript type definitions
│   │   └── pages/           # Page components
│   └── public/              # Static assets
├── docker-compose.yml        # Development orchestration
├── Dockerfile               # Production container
└── README.md               # Project documentation
```

## Performance Guidelines
- **Memory Usage**: Target 4-6GB RAM for full system
- **CPU Optimization**: Designed for 4-core i5 6th gen processors
- **Model Sizes**: Use nano/small variants (YOLOv8n, DistilBERT)
- **Frame Processing**: Skip frames for real-time performance
- **Batch Processing**: Process alerts in batches to reduce overhead

## Security Considerations
- JWT tokens for API authentication
- CORS configuration for frontend access
- Input validation on all API endpoints
- Face redaction for privacy compliance
- Audit logging for data access tracking

## Troubleshooting
- **High CPU Usage**: Reduce model complexity or increase frame skipping
- **Memory Issues**: Enable garbage collection optimization
- **API Timeouts**: Check database connection and query performance
- **Frontend Errors**: Verify CORS settings and API endpoint configuration