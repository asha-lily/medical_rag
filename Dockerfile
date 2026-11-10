# Base image that provides Linux & Python
FROM python:3.12-slim

# Copy the uv program from the official image to the /bin directory of this image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create an /app directory in the image and set it as the working directory
WORKDIR /app

# Create a new environment in /app/.venv and install dependencies from uv.lock
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Add /app/.venv/bin (i.e the .venv's programs) to the front of the PATH list
# so that later commands (python, uvicorn) use the .venv's versions
ENV PATH="/app/.venv/bin:$PATH"

# Download the embedding model into the image, and stop it from contacting Hugging Face when the service starts
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
ENV HF_HUB_OFFLINE=1

# Copy the RAG pipeline code and the pre-built vector store into the image
COPY rag_pipeline/ rag_pipeline/
COPY PIL_docs_vector_store/ PIL_docs_vector_store/

# Define the command to be stored in the image and run each time `docker run` starts a container
# Setting the host to 0.0.0.0 allows the service to accept requests from outside the container
CMD ["uvicorn", "rag_pipeline.api.main:app", "--host", "0.0.0.0", "--port", "8000"]