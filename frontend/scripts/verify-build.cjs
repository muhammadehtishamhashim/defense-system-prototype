#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const DIST_DIR = path.join(__dirname, '..', 'dist');
const REQUIRED_FILES = ['index.html'];
const REQUIRED_DIRS = ['assets'];

console.log('🔍 Verifying production build...');

// Check if dist directory exists
if (!fs.existsSync(DIST_DIR)) {
  console.error('❌ dist directory not found. Run "npm run build" first.');
  process.exit(1);
}

// Check required files
for (const file of REQUIRED_FILES) {
  const filePath = path.join(DIST_DIR, file);
  if (!fs.existsSync(filePath)) {
    console.error(`❌ Required file missing: ${file}`);
    process.exit(1);
  }
  console.log(`✅ Found: ${file}`);
}

// Check required directories
for (const dir of REQUIRED_DIRS) {
  const dirPath = path.join(DIST_DIR, dir);
  if (!fs.existsSync(dirPath)) {
    console.error(`❌ Required directory missing: ${dir}`);
    process.exit(1);
  }
  console.log(`✅ Found: ${dir}/`);
}

// Check index.html content
const indexPath = path.join(DIST_DIR, 'index.html');
const indexContent = fs.readFileSync(indexPath, 'utf8');

// Verify essential elements
const checks = [
  { name: 'Root div', pattern: /<div id="root"><\/div>/ },
  { name: 'Main script', pattern: /<script[^>]*src="[^"]*\.js"[^>]*><\/script>/ },
  { name: 'CSS link', pattern: /<link[^>]*href="[^"]*\.css"[^>]*>/ },
  { name: 'Title', pattern: /<title>.*HifazatAI.*<\/title>/ }
];

for (const check of checks) {
  if (check.pattern.test(indexContent)) {
    console.log(`✅ ${check.name} found in index.html`);
  } else {
    console.warn(`⚠️ ${check.name} not found in index.html`);
  }
}

// Check asset files
const assetsDir = path.join(DIST_DIR, 'assets');
if (fs.existsSync(assetsDir)) {
  const assets = fs.readdirSync(assetsDir);
  const jsFiles = assets.filter(f => f.endsWith('.js'));
  const cssFiles = assets.filter(f => f.endsWith('.css'));
  
  console.log(`✅ Found ${jsFiles.length} JavaScript files`);
  console.log(`✅ Found ${cssFiles.length} CSS files`);
  
  // Check for source maps in production
  const sourceMaps = assets.filter(f => f.endsWith('.map'));
  if (sourceMaps.length > 0) {
    console.log(`ℹ️ Found ${sourceMaps.length} source map files`);
  }
} else {
  console.warn('⚠️ Assets directory not found');
}

// Check file sizes
const getFileSize = (filePath) => {
  const stats = fs.statSync(filePath);
  return (stats.size / 1024).toFixed(2) + ' KB';
};

console.log('\n📊 File sizes:');
console.log(`- index.html: ${getFileSize(indexPath)}`);

if (fs.existsSync(assetsDir)) {
  const assets = fs.readdirSync(assetsDir);
  assets.forEach(asset => {
    const assetPath = path.join(assetsDir, asset);
    if (fs.statSync(assetPath).isFile()) {
      console.log(`- assets/${asset}: ${getFileSize(assetPath)}`);
    }
  });
}

console.log('\n✅ Build verification completed successfully!');
console.log('\n🚀 To test the build locally, run:');
console.log('   npm run preview');
console.log('   or');
console.log('   npm run serve');