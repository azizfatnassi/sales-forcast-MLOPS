#!/usr/bin/env bash
set -e

echo "Starting API container..."
docker run -d -p 8001:8001 --name test-api sales-forcast-api:ci

sleep 10

echo "Calling /predict endpoint..."
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
  -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"data":[{"Store":1,"DayOfWeek":5,"Open":1,"Promo":0}]}')

echo "HTTP status: $RESPONSE"

docker stop test-api
docker rm test-api

if [ "$RESPONSE" -ne 200 ]; then
  echo "API test failed!"
  exit 1
fi

echo "API test passed"
