# ───────────────────────────────────────────────────────────
# Stage 1: Build dependencies
# ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# ensure non‑interactive installs
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install any system build‑deps (e.g. for wheel building)
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc \
 && rm -rf /var/lib/apt/lists/*

# Copy only what's needed to install Python deps
COPY requirements.txt ./
COPY BasicSR ./BasicSR

# Install Python dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ───────────────────────────────────────────────────────────
# Stage 2: Final image
# ───────────────────────────────────────────────────────────
FROM python:3.11-slim

# ensure non‑interactive installs
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install OpenCV runtime deps (libGL, etc.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1-mesa-glx \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages & CLI entrypoints
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app.py ./

# Copy the model file(s)
COPY models ./models

# Expose the port FastAPI/uvicorn will run on
EXPOSE 8000

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
