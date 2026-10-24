import kagglehub
from pathlib import Path

# kagglehub.login()

path = Path.cwd() / "dataset"
path.mkdir(exist_ok=True)
# Download latest version
path = kagglehub.competition_download('house-prices-advanced-regression-techniques', output_dir=path)

print("Path to competition files:", path)