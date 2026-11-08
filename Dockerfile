# The public demo site: the app in its simulation mode, open to visitors. It never reaches a robot.
#   docker build -t dream-chess . && docker run -p 7860:7860 dream-chess
# Hosted on Hugging Face Spaces (Docker); .github/workflows/deploy-space.yml pushes each change on main there.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends stockfish libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ENV STOCKFISH_PATH=/usr/games/stockfish PYTHONUNBUFFERED=1

# Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
RUN pip install --no-cache-dir "chess>=1.10" "numpy>=1.26" "opencv-python-headless>=4.8" "pillow>=10.1"
COPY --chown=user . .
USER user

EXPOSE 7860
CMD ["python", "app.py", "--public", "--port", "7860", "--pace", "1.2", "--sim-success", "0.6", "--log-dir", "/tmp/games"]
