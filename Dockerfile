# Snake AI lab in a container (Linux, CPU only).
#
#   docker build -t snake-ai .
#   docker run --rm -p 8000:8000 -v "$PWD/lessons:/app/lessons" -v "$PWD/ideas:/app/ideas" snake-ai
#   then open http://localhost:8000
#
# The volumes keep your runs and ideas outside the container. For the AI analysis add --env-file .env.
# The planner reads the container's CPU and memory limits, e.g. docker run --cpus=4 --memory=6g ...
#
#   docker run --rm snake-ai python -m pytest -q tests      # run the tests
ARG PYTHON=3.12
FROM python:${PYTHON}-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# CPU-only PyTorch is much smaller than the default build and is all this project needs
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
 && pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "-m", "dashboard.server", "--host", "0.0.0.0", "--port", "8000"]
