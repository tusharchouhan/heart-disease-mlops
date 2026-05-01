# MLOPS
MLOPS Assignment
## Cloud Deployment

**Platform:** Google Cloud Run
**URL:** https://heart-disease-api-xxxxxxxxxx-uc.a.run.app
**Region:** us-central1
**Continuous Deployment:** Enabled (auto-deploys on git push to main)

### Test the Live API
```bash
# Health check
curl https://heart-disease-api-xxxxxxxxxx-uc.a.run.app/health

# Prediction
curl -X POST https://heart-disease-api-xxxxxxxxxx-uc.a.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"age":55,"sex":1,"cp":2,"trestbps":130,"chol":250,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.5,"slope":1,"ca":0,"thal":3}'
