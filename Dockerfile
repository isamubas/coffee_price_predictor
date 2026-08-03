# Hugging Face Spaces no longer offers Streamlit as a native SDK (only gradio,
# docker, static), so the app ships as a Docker Space instead. Spaces expect
# the app on port 7860.
FROM python:3.11-slim

# Spaces run containers as UID 1000; create a matching user so Streamlit can
# write its config and cache.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

# Streamlit's built-in health endpoint. Uses python rather than curl, which
# python:3.11-slim does not ship.
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health')"

CMD ["streamlit", "run", "app.py"]
