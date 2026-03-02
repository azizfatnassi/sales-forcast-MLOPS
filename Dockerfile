# 1️⃣ Use lightweight Python 3.10 image
FROM python:3.10-slim

# 2️⃣ Set working directory inside container
WORKDIR /app

# 3️⃣ Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 4️⃣ Build argument for model version
ARG VERSION=dev
ENV VERSION=${VERSION}

# 5️⃣ Copy the versioned model and features dynamically
COPY models/model_${VERSION}.pkl models/model_${VERSION}.pkl
COPY models/features.json models/features.json

# 6️⃣ Copy the rest of the project (API code, etc.)
COPY . .

# 7️⃣ Expose the port FastAPI will use
EXPOSE 8001

# 8️⃣ Run FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]