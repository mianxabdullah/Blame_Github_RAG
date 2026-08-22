from pathlib import Path
from git import Repo

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".vue", ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".cs", ".swift", ".kt",
    ".md", ".yaml", ".yml", ".json", ".sql", ".html", ".css", ".sh", ".ipynb",
    ".toml", ".xml", ".scss", ".sass", ".less", ".tf", ".gradle", ".kts",
    ".scala", ".dart", ".lua", ".r",
}

def clone_repo(repo_url: str, clone_dir: str = "data/repos") -> Path:
    """Download a GitHub repo to a local folder."""
    # Turn a URL like "https://github.com/psf/requests" into "requests"
    """
    repo_url.rstrip("/").split("/")[-1] — GitHub URLs look like https://github.com/psf/requests. 
    Splitting on / gives ['https:', '', 'github.com', 'psf', 'requests'], and [-1] grabs the last piece 
    — the repo's actual name. 
    .rstrip("/") just guards against a trailing slash (.../requests/) messing up that split.
    """
    repo_name = repo_url.rstrip("/").split("/")[-1]

    dest = Path(clone_dir) / repo_name 

    if dest.exists():
        print(f"Repo already downloaded at {dest}")
    else:
        print(f"Cloning {repo_url}...")
        Repo.clone_from(repo_url, str(dest)) # this is the actual download step, from the GitPython library. It does the same thing as running git clone <url> <folder> in a terminal, just from Python.

    return dest