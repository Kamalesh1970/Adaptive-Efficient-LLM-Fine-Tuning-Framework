# ==============================================================================
# Adaptive LLM Fine-Tuning Studio - Frontend Container
# ==============================================================================

# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci --prefer-offline || npm install

COPY . .
RUN npm run build

# Production runtime stage
FROM nginx:1.27-alpine AS runner

# Remove default nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy built static assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Healthcheck
HEALTHCHECK --interval=20s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://localhost:80/healthz || exit 1

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
