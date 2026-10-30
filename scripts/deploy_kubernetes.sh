#!/usr/bin/env bash
# ==============================================================================
# Adaptive LLM Fine-Tuning Studio - Local Kubernetes Deployment & Verification
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST_DIR="${ROOT_DIR}/deploy/kubernetes"
NAMESPACE="adaptive-studio"

ACTION="${1:-deploy}"

if [[ "${ACTION}" == "cleanup" || "${ACTION}" == "delete" ]]; then
    echo "=== Cleaning up Adaptive LLM Fine-Tuning Studio Kubernetes Workloads ==="
    if kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
        echo "Deleting namespace ${NAMESPACE}..."
        kubectl delete namespace "${NAMESPACE}" --timeout=60s || true
        echo "Cleanup complete."
    else
        echo "Namespace ${NAMESPACE} not found. Nothing to clean up."
    fi
    exit 0
fi

echo "========================================================================"
echo "Adaptive LLM Fine-Tuning Studio: Kubernetes Deployment"
echo "========================================================================"

# 1. Verify kubectl connectivity
if ! command -v kubectl &> /dev/null; then
    echo "[ERROR] kubectl is not installed or not in PATH."
    echo "Install kubectl or run in a cluster-enabled environment."
    exit 1
fi

echo "[INFO] Current Kubernetes Context: $(kubectl config current-context 2>/dev/null || echo 'None')"

# 2. Deploy Namespace and Prerequisites
echo "[1/6] Applying Namespace and RBAC/Secrets..."
kubectl apply -f "${MANIFEST_DIR}/namespace.yaml"
kubectl apply -f "${MANIFEST_DIR}/secret.yaml"
kubectl apply -f "${MANIFEST_DIR}/configmap.yaml"
kubectl apply -f "${MANIFEST_DIR}/pvc.yaml"
kubectl apply -f "${MANIFEST_DIR}/network-policy.yaml"

# 3. Deploy Backend and Services
echo "[2/6] Deploying Backend API and Service..."
kubectl apply -f "${MANIFEST_DIR}/backend-deployment.yaml"
kubectl apply -f "${MANIFEST_DIR}/backend-service.yaml"

# 4. Deploy Frontend and Services
echo "[3/6] Deploying Frontend Web UI and Service..."
kubectl apply -f "${MANIFEST_DIR}/frontend-deployment.yaml"
kubectl apply -f "${MANIFEST_DIR}/frontend-service.yaml"

# 5. Apply Ingress
echo "[4/6] Applying Ingress Configuration..."
kubectl apply -f "${MANIFEST_DIR}/ingress.yaml"

# 6. Wait for Rollout
echo "[5/6] Waiting for workload rollout..."
kubectl rollout status deployment/studio-backend -n "${NAMESPACE}" --timeout=90s || true
kubectl rollout status deployment/studio-frontend -n "${NAMESPACE}" --timeout=90s || true

# 7. Print Cluster Status
echo "[6/6] Workload Status in Namespace '${NAMESPACE}':"
kubectl get pods,services,pvc -n "${NAMESPACE}" -o wide

echo "========================================================================"
echo "Deployment verification script finished."
echo "To clean up all deployed resources, run: $0 cleanup"
echo "========================================================================"
