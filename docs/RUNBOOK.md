# Runbook

## Where to look
- GET /health : status ok or degraded
- GET /metrics : requests_total, error_rate, avg_latency_ms, p95_latency_ms
- logs/app.log : application log
- logs/predictions.jsonl : every prediction (input + output)

## Alerts and actions
1. error_rate above 0.05 : read logs/app.log for Prediction failed, check the last inputs in logs/predictions.jsonl, fix the bad input or roll back the last commit.
2. p95_latency_ms above 500 : check container CPU and memory (docker stats), then restart with docker compose restart.
3. /health returns degraded or 503 : model files missing or unreadable. Check models/model.pkl, models/state_encoder.pkl, models/feature_list.txt, then docker compose up --build.
4. Drift suspected (probability values shifting) : compare recent inputs in logs/predictions.jsonl with the training data, and retrain if the distribution changed.
