# 🚀 Bank Marketing Campaign Prediction — End-to-End Kubernetes Deployment Guide

A complete, step-by-step guide to containerizing the **Bank Marketing Campaign Prediction** Streamlit application, loading the Docker image into a local Docker Desktop Kubernetes cluster, deploying multi-replica Pods, exposing the service, and managing version control cleanly.

---

## 🏗️ 1. System Architecture & Traffic Flow

Below is how requests travel from your browser through Kubernetes to your machine learning model (`models/model.pkl`):

```text
+-----------------------------------------------------------------------------------+
|                               LOCAL WORKSTATION                                   |
|                                                                                   |
|   [ Web Browser ]                                                                 |
|    http://localhost:8501  (via kubectl port-forward)                              |
|    http://localhost:30085 (via NodePort)                                          |
|          |                                                                        |
|          v                                                                        |
|   +---------------------------------------------------------------------------+   |
|   |               DOCKER DESKTOP KUBERNETES CLUSTER (desktop-control-plane)   |   |
|   |                                                                           |   |
|   |   +-------------------------------------------------------------------+   |   |
|   |   |  Kubernetes Service: bank-marketing-service (NodePort)            |   |   |
|   |   |  ClusterIP: 10.96.185.244 | Port: 8501 -> TargetPort: 8501        |   |   |
|   |   +-------------------------------------------------------------------+   |   |
|   |                     |                               |                     |   |
|   |          (Load Balances Traffic)         (Load Balances Traffic)          |   |
|   |                     v                               v                     |   |
|   |   +---------------------------------+   +-----------------------------+   |   |
|   |   | Pod 1: bank-marketing-predictor |   | Pod 2: bank-marketing-...   |   |   |
|   |   | IP: 10.244.0.7:8501             |   | IP: 10.244.0.8:8501         |   |   |
|   |   |                                 |   |                             |   |   |
|   |   |  +---------------------------+  |   |  +-----------------------+  |   |   |
|   |   |  | Container: streamlit-app  |  |   |  | Container: streamlit  |  |   |   |
|   |   |  | Image: bank-ml-app:latest |  |   |  | Image: bank-ml-app... |  |   |   |
|   |   |  | - app.py                  |  |   |  | - app.py              |  |   |   |
|   |   |  | - models/model.pkl        |  |   |  | - models/model.pkl    |  |   |   |
|   |   |  | - model_comparison.csv    |  |   |  | - model_comparison.csv|  |   |   |
|   |   |  +---------------------------+  |   |  +-----------------------+  |   |   |
|   |   +---------------------------------+   +-----------------------------+   |   |
|   +---------------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------------+
```

### Key Components Explained
| Component | Name | Role |
| :--- | :--- | :--- |
| **Docker Image** | `bank-ml-app:latest` | Self-contained Python 3.11 environment bundling `app.py`, dependencies, `models/model.pkl`, and `model_comparison.csv`. |
| **Deployment** | `bank-marketing-predictor` | Ensures **2 replicas (Pods)** are always running, enforces CPU/Memory limits, and monitors health via `/_stcore/health`. |
| **Service** | `bank-marketing-service` | Provides a single, stable network endpoint (`8501`) that distributes incoming requests across both running Pods. |

---

## 📁 2. Project Directory Structure

Ensure your project folder (`ml flow`) is structured cleanly before deploying:

```text
ml flow/
├── k8s/
│   ├── deployment.yaml          # Kubernetes Deployment manifest (2 replicas + health probes)
│   └── service.yaml             # Kubernetes Service manifest (NodePort 30085 -> 8501)
├── models/
│   ├── model.pkl                # Champion trained ML model (410 KB)
│   ├── DecisionTree.pkl
│   └── LogisticRegression.pkl
├── app.py                       # Streamlit web application
├── bank_mlflow_all_models.py    # MLflow training & evaluation script
├── model_comparison.csv         # Model evaluation metrics
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container build instructions
└── .gitignore                   # Excludes myenv/, mlruns/, *.tar, etc.
```

---

## 🛠️ 3. Step-by-Step Deployment Workflow

### Step 1: Verify Prerequisites & Enable Kubernetes
1. Open **Docker Desktop** on Windows.
2. Navigate to **Settings (⚙️) → Kubernetes**, check **Enable Kubernetes**, and click **Apply & restart**.
3. Open a terminal in your `ml flow` directory and verify that `kubectl` connects to your local cluster:

```cmd
kubectl config use-context docker-desktop
kubectl cluster-info
kubectl get nodes
```
* **Expected Output:** `desktop-control-plane` shows `STATUS` as `Ready`.

---

### Step 2: Build the Docker Image
Ensure your `Dockerfile` packages the Streamlit app and trained model:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models/ ./models/
COPY model_comparison.csv .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build the image locally:

```cmd
docker build -t bank-ml-app:latest .
```

---

### Step 3: Import the Local Image Into Kubernetes (`k8s.io` Namespace)
Docker Desktop stores standard CLI images separately from the Kubernetes containerd store (`k8s.io`). Without importing, Pods configured with `imagePullPolicy: Never` will fail with `ErrImageNeverPull`.

```text
[Docker CLI Store] ---> docker save ---> [bank-ml-app.tar] ---> ctr import ---> [Kubernetes k8s.io Store]
```

1. **Export the image to a `.tar` archive:**
   ```cmd
   docker save -o bank-ml-app.tar bank-ml-app:latest
   ```
2. **Import the archive into the Kubernetes control plane:**
   ```cmd
   docker exec -i desktop-control-plane ctr -n k8s.io images import - < bank-ml-app.tar
   ```
3. **Delete the temporary `.tar` file immediately** (prevents accidental Git pushes of >100 MB files):
   ```cmd
   del bank-ml-app.tar
   ```

---

### Step 4: Configure Kubernetes Manifests

#### 1. `k8s/deployment.yaml`
Defines 2 replicas, resource requests/limits, and Streamlit health probes:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bank-marketing-predictor
  labels:
    app: bank-marketing-predictor
spec:
  replicas: 2
  selector:
    matchLabels:
      app: bank-marketing-predictor
  template:
    metadata:
      labels:
        app: bank-marketing-predictor
    spec:
      containers:
        - name: streamlit-app
          image: bank-ml-app:latest
          imagePullPolicy: Never
          ports:
            - containerPort: 8501
          resources:
            requests:
              cpu: "250m"
              memory: "512Mi"
            limits:
              cpu: "1000m"
              memory: "1Gi"
          readinessProbe:
            httpGet:
              path: /_stcore/health
              port: 8501
            initialDelaySeconds: 10
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /_stcore/health
              port: 8501
            initialDelaySeconds: 20
            periodSeconds: 15
```

#### 2. `k8s/service.yaml`
Exposes the Pods via a `NodePort` service:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: bank-marketing-service
spec:
  type: NodePort
  selector:
    app: bank-marketing-predictor
  ports:
    - protocol: TCP
      port: 8501
      targetPort: 8501
      nodePort: 30085
```

---

### Step 5: Deploy & Verify Cluster Health
Apply both manifests to your cluster:

```cmd
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Verify that all Kubernetes objects are healthy:

```cmd
# 1. Check Deployment status (Should show 2/2 READY)
kubectl get deployment bank-marketing-predictor

# 2. Check Pod status (Both Pods should show 1/1 Running)
kubectl get pods -l app=bank-marketing-predictor

# 3. Check Service & Pod Endpoints
kubectl get svc bank-marketing-service
kubectl get endpoints bank-marketing-service
```

---

### Step 6: Access the Application
Forward local port `8501` directly to your Kubernetes Service:

```cmd
kubectl port-forward svc/bank-marketing-service 8501:8501
```
Keep this terminal window running and open **`http://localhost:8501`** in your web browser.
* Test single predictions in the **Predict** tab.
* Upload batch CSV files in the **Batch** tab.
* Review model accuracy/metrics in the **Model performance** tab.

---

## 🔄 4. Day-2 Operations: Updating, Scaling & Cleanup

### Updating the Model or App Code
Whenever you retrain `models/model.pkl` or edit `app.py`:
1. Rebuild the image: `docker build -t bank-ml-app:latest .`
2. Export, import into `k8s.io`, and remove the `.tar`:
   ```cmd
   docker save -o bank-ml-app.tar bank-ml-app:latest
   docker exec -i desktop-control-plane ctr -n k8s.io images import - < bank-ml-app.tar
   del bank-ml-app.tar
   ```
3. Perform a zero-downtime rolling restart:
   ```cmd
   kubectl rollout restart deployment/bank-marketing-predictor
   ```

### Scaling Replicas
Increase or decrease the number of running Pods dynamically:
```cmd
kubectl scale deployment bank-marketing-predictor --replicas=3
```

### Removing the Deployment
To stop port-forwarding, press `Ctrl + C` in the active terminal, then delete the cluster resources:
```cmd
kubectl delete -f k8s/service.yaml
kubectl delete -f k8s/deployment.yaml
```

---

## 🚨 5. Troubleshooting Quick Reference

| Error / Symptom | Root Cause | Resolution |
| :--- | :--- | :--- |
| `Unable to connect to the server: localhost:8080` | Kubernetes cluster is not running or `kubectl` context is unset. | Start Kubernetes in Docker Desktop and run `kubectl config use-context docker-desktop`. |
| `ErrImageNeverPull` on Pods | `bank-ml-app:latest` is not in Kubernetes' `k8s.io` image store. | Export via `docker save` and import using `ctr -n k8s.io images import` (Step 3). |
| `ERR_CONNECTION_REFUSED` on `localhost:30085` | Windows/Docker Desktop networking blocks direct `NodePort` routing. | Run `kubectl port-forward svc/bank-marketing-service 8501:8501` and visit `http://localhost:8501`. |
| GitHub `GH001: Large files detected` | A `.tar` image archive (>100 MB) was staged in Git. | Delete the `.tar` file, add `*.tar` to `.gitignore`, undo the commit (`git reset --soft HEAD~1`), and re-commit. |
