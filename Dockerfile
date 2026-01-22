# 1 Use lightweight Python 3.10 image
FROM python:3.10-slim

#Set working directory inside container
WORKDIR /app
# Copy requirementss.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

#4 Copy the rest of the project ( code+ model + features) 
COPY . .

#5 Tell Docker the port th API will use 
EXPOSE 8001
# Run the FastAPI API 
CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8001"]