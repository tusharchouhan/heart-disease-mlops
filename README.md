# MLOPS
MLOPS Assignment
## Cloud Deployment

**Platform:** Google Cloud Run
**URL:** https://heart-disease-mlops-951222900001.us-central1.run.app
**Region:** us-central1
**Continuous Deployment:** Enabled (auto-deploys on git push to main)

### Test the Live API
```bash
# Health check
curl https://heart-disease-mlops-951222900001.europe-west1.run.app/health

# Prediction
curl -X POST https://heart-disease-mlops-951222900001.europe-west1.run.app/predict 
  -H "Content-Type: application/json" \
  -d '{"age":55,"sex":1,"cp":2,"trestbps":130,"chol":250,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.5,"slope":1,"ca":0,"thal":3}'
