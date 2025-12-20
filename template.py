import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

project_name = "SmartRouteAI"

list_of_files = [

    # GitHub CI
    ".github/workflows/ci.yaml",

    # Configs
    "config/config.yaml",
    "config/models.yaml",
    "config/routing.yaml",

    # Data
    "data/documents/.gitkeep",
    "data/embeddings/.gitkeep",
    "data/queries/.gitkeep",
    "data/costs/.gitkeep",

    # Models
    "models/local/.gitkeep",
    "models/classifier/.gitkeep",

    # Source package
    f"src/{project_name}/__init__.py",

    # src/models
    f"src/{project_name}/models/__init__.py",
    f"src/{project_name}/models/local_model.py",
    f"src/{project_name}/models/api_model.py",
    f"src/{project_name}/models/model_manager.py",

    # src/routing
    f"src/{project_name}/routing/__init__.py",
    f"src/{project_name}/routing/classifier.py",
    f"src/{project_name}/routing/router.py",
    f"src/{project_name}/routing/features.py",

    # src/retrieval
    f"src/{project_name}/retrieval/__init__.py",
    f"src/{project_name}/retrieval/embedder.py",
    f"src/{project_name}/retrieval/retriever.py",

    # src/cost
    f"src/{project_name}/cost/__init__.py",
    f"src/{project_name}/cost/tracker.py",
    f"src/{project_name}/cost/budget.py",
    f"src/{project_name}/cost/reporter.py",

    # src/evaluation
    f"src/{project_name}/evaluation/__init__.py",
    f"src/{project_name}/evaluation/evaluator.py",
    f"src/{project_name}/evaluation/metrics.py",

    # src/pipeline
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training.py",
    f"src/{project_name}/pipeline/inference.py",
    f"src/{project_name}/pipeline/evaluation.py",

    # src/api
    f"src/{project_name}/api/__init__.py",
    f"src/{project_name}/api/main.py",
    f"src/{project_name}/api/routes/query.py",
    f"src/{project_name}/api/routes/metrics.py",

    # src/utils
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/config.py",
    f"src/{project_name}/utils/logging.py",
    f"src/{project_name}/utils/helpers.py",

    # Dashboard
    "dashboard/app.py",
    "dashboard/pages/cost_analytics.py",
    "dashboard/pages/model_performance.py",
    "dashboard/pages/query_logs.py",

    # Scripts
    "scripts/setup_models.py",
    "scripts/train_classifier.py",
    "scripts/evaluate.py",
    "scripts/export_report.py",

    # Tests
    "tests/test_router.py",
    "tests/test_cost_tracker.py",
    "tests/test_api.py",

    # Notebooks
    "notebooks/01_data_exploration.ipynb",
    "notebooks/02_classifier_training.ipynb",
    "notebooks/03_cost_analysis.ipynb",

    # Docker
    "docker/Dockerfile",
    "docker/docker-compose.yml",

    # Root files
    "requirements.txt",
]


for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)

    if file_dir:
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Created directory: {file_dir}")

    if not file_path.exists():
        with open(file_path, "w") as f:
            pass
        logging.info(f"Created file: {file_path}")
    else:
        logging.info(f"File already exists: {file_path}")
