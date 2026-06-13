# ---- Build stage ----
FROM node:22-alpine AS build
WORKDIR /app

# Install all deps (incl. dev) for the Nest build
COPY package*.json ./
RUN npm ci

# Compile TypeScript -> dist
COPY tsconfig.json nest-cli.json ./
COPY src ./src
RUN npm run build

# ---- Runtime stage ----
FROM node:22-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production

# Production dependencies only
COPY package*.json ./
RUN npm ci --omit=dev && npm cache clean --force

# Compiled app
COPY --from=build /app/dist ./dist

# Storage lives on a mounted volume (see docker-compose)
ENV STORAGE_DIR=/data
RUN mkdir -p /data

EXPOSE 3001
CMD ["node", "dist/main.js"]
