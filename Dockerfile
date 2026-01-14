# Multi-stage Dockerfile for CF Travel Bot
# Stage 1: Build application (Frontend & Backend)
FROM node:20-alpine AS app-builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install build dependencies
RUN apk add --no-cache python3 make g++

# Install all dependencies (including devDependencies for build)
RUN npm ci

# Copy source code
COPY . .

# Build the application (runs build:server and vite build)
RUN npm run build

# Stage 2: Production image
FROM node:20-alpine AS production

WORKDIR /app

# Install dumb-init for proper signal handling
RUN apk add --no-cache dumb-init

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

# Copy package files
COPY package*.json ./

# Install build dependencies
RUN apk add --no-cache python3 make g++

# Install production dependencies only
RUN npm ci --omit=dev && npm cache clean --force

# Copy built backend and frontend from builder stage
COPY --from=app-builder /app/dist-server ./dist-server
COPY --from=app-builder /app/dist ./dist

# Create log and data directories
RUN mkdir -p /var/log/cbthis /app/dist-server/data && \
    chown -R nodejs:nodejs /var/log/cbthis /app/dist-server/data

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3000

# Switch to non-root user
USER nodejs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000/health', (r) => process.exit(r.statusCode === 200 ? 0 : 1)).on('error', () => process.exit(1))"

# Start the application
ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist-server/main.js"]
