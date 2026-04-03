import importlib.util
import os

import uvicorn


def load_backend_app():
    """
    Load `rag-research/main.py` directly from its file path.
    (The `rag-research/` folder name contains a hyphen, so it can't be imported normally.)
    """
    here = os.path.dirname(__file__)
    backend_path = os.path.join(here, "rag-research", "main.py")

    spec = importlib.util.spec_from_file_location("rag_backend_main", backend_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load backend module from: {backend_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module.app


if __name__ == "__main__":
    app = load_backend_app()
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

import importlib.util
import os

import uvicorn


def load_backend_app():
    """
    Load `rag-research/main.py` without treating `rag-research/` as a Python package
    (the folder name contains a hyphen).
    """
    here = os.path.dirname(__file__)
    backend_path = os.path.join(here, "rag-research", "main.py")

    spec = importlib.util.spec_from_file_location("rag_backend_main", backend_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load backend module from: {backend_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module.app


if __name__ == "__main__":
    app = load_backend_app()
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

