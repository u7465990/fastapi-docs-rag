"""Console entry point for the API server."""

from __future__ import annotations

import uvicorn

from fastapi_docs_rag.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "fastapi_docs_rag.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "dev",
    )


if __name__ == "__main__":
    main()
