# Production Deployment Guide

## Overview

This guide covers deploying the HifazatAI frontend to production environments.

## Build Process

### 1. Environment Configuration

Create appropriate environment files:

```bash
# .env.production - Production environment variables
VITE_API_URL=https://your-api-domain.com
VITE_NODE_ENV=production
VITE_BUILD_TARGET=production
VITE_ENABLE_DEBUG=false
VITE_ENABLE_SOURCEMAPS=false
VITE_ENABLE_LAZY_LOADING=true
```

### 2. Build Commands

```bash
# Standard build
npm run build

# Build with verification
npm run build:verify

# Preview build locally
npm run preview

# Serve production build
npm run serve
```

### 3. Build Output

The build process creates:
- `dist/index.html` - Main HTML file
- `dist/assets/` - JavaScript, CSS, and other assets
- `dist/vite.svg` - Favicon

## Deployment Options

### Option 1: Static File Hosting

Deploy the `dist/` folder to any static hosting service:

- **Netlify**: Drag and drop the `dist` folder
- **Vercel**: Connect your Git repository
- **AWS S3**: Upload files to S3 bucket with static hosting
- **GitHub Pages**: Use GitHub Actions to deploy

### Option 2: Node.js Server

Use the included production server:

```bash
# Install dependencies
npm install --production

# Build the application
npm run build

# Start production server
npm run serve
```

### Option 3: Docker Deployment

```dockerfile
# Multi-stage build
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Option 4: Cloud Platforms

#### Google Cloud Platform
```bash
# Build for production
npm run build

# Deploy to Cloud Storage
gsutil -m cp -r dist/* gs://your-bucket-name/

# Or deploy to Cloud Run
gcloud run deploy hifazat-frontend --source .
```

#### AWS
```bash
# Build for production
npm run build

# Deploy to S3
aws s3 sync dist/ s3://your-bucket-name/

# Or use AWS Amplify
amplify publish
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API URL | `http://localhost:8000` |
| `VITE_NODE_ENV` | Environment mode | `development` |
| `VITE_ENABLE_DEBUG` | Enable debug logging | `false` |
| `VITE_ENABLE_SOURCEMAPS` | Include source maps | `false` |

### Base URL Configuration

For subdirectory deployments, update `vite.config.ts`:

```typescript
export default defineConfig({
  base: '/your-subdirectory/',
  // ... other config
})
```

### CORS Configuration

Ensure your backend allows requests from your frontend domain:

```python
# FastAPI backend
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Performance Optimization

### 1. Asset Optimization

The build automatically:
- Minifies JavaScript and CSS
- Optimizes images
- Generates efficient chunk splitting
- Enables gzip compression

### 2. Caching Strategy

Configure your web server for optimal caching:

```nginx
# nginx.conf
location /assets/ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}

location / {
    try_files $uri $uri/ /index.html;
    expires -1;
    add_header Cache-Control "no-cache, no-store, must-revalidate";
}
```

### 3. CDN Integration

For better performance, serve assets from a CDN:

```typescript
// vite.config.ts
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        assetFileNames: 'assets/[name]-[hash].[ext]'
      }
    }
  }
})
```

## Monitoring and Debugging

### 1. Error Tracking

The application includes comprehensive error handling:
- Global error boundaries
- Unhandled promise rejection handling
- Error logging to localStorage
- Production error reporting

### 2. Performance Monitoring

Monitor key metrics:
- Page load times
- API response times
- JavaScript errors
- User interactions

### 3. Health Checks

The application provides health check endpoints:
- `/` - Main application
- API connectivity check
- Real-time status monitoring

## Troubleshooting

### Common Issues

1. **Blank Page After Deployment**
   - Check browser console for errors
   - Verify API URL configuration
   - Ensure proper base URL setting

2. **API Connection Errors**
   - Verify CORS configuration
   - Check network connectivity
   - Validate API endpoint URLs

3. **Asset Loading Issues**
   - Check file paths in built HTML
   - Verify web server configuration
   - Ensure proper MIME types

### Debug Mode

Enable debug mode for troubleshooting:

```bash
# .env.local
VITE_ENABLE_DEBUG=true
VITE_ENABLE_SOURCEMAPS=true
```

### Log Analysis

Check browser console for:
- Application startup logs
- API configuration
- Error details with IDs
- Performance metrics

## Security Considerations

### 1. Environment Variables

- Never expose sensitive data in VITE_ variables
- Use server-side configuration for secrets
- Validate all environment inputs

### 2. Content Security Policy

Implement CSP headers:

```html
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';">
```

### 3. HTTPS Configuration

Always use HTTPS in production:
- Configure SSL certificates
- Enable HSTS headers
- Redirect HTTP to HTTPS

## Maintenance

### 1. Updates

Regular maintenance tasks:
- Update dependencies
- Security patches
- Performance optimizations
- Feature updates

### 2. Backup Strategy

Backup important data:
- Source code (Git repository)
- Environment configurations
- Build artifacts
- User data (if applicable)

### 3. Rollback Plan

Prepare for rollbacks:
- Keep previous build artifacts
- Document configuration changes
- Test rollback procedures
- Monitor after deployments

## Support

For deployment issues:
1. Check this guide first
2. Review browser console logs
3. Verify environment configuration
4. Test with local production build
5. Contact development team with error details