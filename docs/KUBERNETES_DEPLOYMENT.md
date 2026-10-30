# Kubernetes Deployment & Infrastructure Guide

This guide details the deployment of **Adaptive LLM Fine-Tuning Studio** onto a local or production Kubernetes cluster using the verified declarative manifests in `deploy/kubernetes/`.

---

## 1. Architectural Overview

The deployment consists of 12 production-grade declarative manifests located in `deploy/kubernetes/`:

| Manifest | Kind | Description |
|---|---|---|
| `namespace.yaml` | `Namespace` | Dedicated `adaptive-studio` namespace with security isolation. |
| `secret.yaml` | `Secret` | Sensitive configuration (Hugging Face tokens, API keys). |
| `configmap.yaml` | `ConfigMap` | Non-sensitive runtime variables (hosts, ports, log levels). |
| `pvc.yaml` | `PersistentVolumeClaim` | 50Gi ReadWriteOnce persistent storage for models & checkpoints. |
| `backend-deployment.yaml` | `Deployment` | FastAPI backend with liveness/readiness probes & resource limits. |
| `backend-service.yaml` | `Service` | ClusterIP service exposing port 8000 for backend API. |
| `frontend-deployment.yaml` | `Deployment` | Nginx React frontend with `/healthz` endpoint. |
| `frontend-service.yaml` | `Service` | ClusterIP service exposing port 80 for UI traffic. |
| `ingress.yaml` | `Ingress` | Nginx Ingress routing `/api` to backend and `/` to frontend. |
| `network-policy.yaml` | `NetworkPolicy` | Ingress and egress traffic isolation between pods. |
| `training-job.yaml` | `Job` | CPU fine-tuning job template with bounded execution. |
| `gpu-training-job.yaml` | `Job` | GPU fine-tuning job requiring `nvidia.com/gpu: 1` resource limits. |

---

## 2. Local Cluster Setup (Kind / Minikube)

### Option A: Kind (Kubernetes in Docker)
```bash
# 1. Create a Kind cluster with local port mappings
cat <<EOF | kind create cluster --name studio-cluster --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  extraPortMappings:
  - containerPort: 80
    hostPort: 8080
    protocol: TCP
EOF

# 2. Deploy the Studio stack
./scripts/deploy_kubernetes.sh deploy
```

### Option B: Minikube
```bash
# 1. Start Minikube with Docker driver
minikube start --driver=docker --memory=2048 --cpus=2

# 2. Deploy the Studio stack
./scripts/deploy_kubernetes.sh deploy
```

---

## 3. Health & Readiness Verification

Run the following diagnostics to confirm healthy operation:

```bash
# Inspect pod readiness in the namespace
kubectl get pods -n adaptive-studio -o wide

# Check service discovery
kubectl get svc -n adaptive-studio

# Check persistent volume binding
kubectl get pvc -n adaptive-studio

# Verify backend health endpoint via port-forwarding
kubectl port-forward svc/studio-backend-service 8000:8000 -n adaptive-studio &
curl http://localhost:8000/api/health
```

---

## 4. Teardown and Cleanup

To completely remove the deployment and all allocated resources:

```bash
./scripts/deploy_kubernetes.sh cleanup
# If using Kind:
kind delete cluster --name studio-cluster
```
