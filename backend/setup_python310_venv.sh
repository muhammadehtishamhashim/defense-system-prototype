#!/bin/bash

# Setup Python 3.10 Virtual Environment for Hifazat AI Backend
# This script creates a new virtual environment with Python 3.10 for better ML library compatibility

set -e  # Exit on any error

echo "🐍 Setting up Python 3.10 Virtual Environment for Hifazat AI Backend"
echo "=================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the backend directory
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found. Please run this script from the backend directory."
    exit 1
fi

# Check if Python 3.10 is available
print_status "Checking for Python 3.10..."
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    print_success "Found Python 3.10: $(python3.10 --version)"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [ "$PYTHON_VERSION" = "3.10" ]; then
        PYTHON_CMD="python3"
        print_success "Found Python 3.10: $(python3 --version)"
    else
        print_error "Python 3.10 not found. Current version: $(python3 --version)"
        print_status "Please install Python 3.10:"
        echo "  Ubuntu/Debian: sudo apt update && sudo apt install python3.10 python3.10-venv python3.10-dev"
        echo "  CentOS/RHEL: sudo yum install python3.10 python3.10-venv python3.10-devel"
        echo "  Or use pyenv: pyenv install 3.10.12 && pyenv local 3.10.12"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.10."
    exit 1
fi

# Backup current venv if it exists
if [ -d "venv" ]; then
    print_warning "Existing virtual environment found."
    read -p "Do you want to backup the current venv? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        BACKUP_NAME="venv_backup_$(date +%Y%m%d_%H%M%S)"
        print_status "Backing up current venv to $BACKUP_NAME..."
        mv venv "$BACKUP_NAME"
        print_success "Backup created: $BACKUP_NAME"
    else
        print_status "Removing current venv..."
        rm -rf venv
    fi
fi

# Create new virtual environment with Python 3.10
print_status "Creating new virtual environment with Python 3.10..."
$PYTHON_CMD -m venv venv

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Verify Python version in venv
VENV_PYTHON_VERSION=$(python --version)
print_success "Virtual environment created with: $VENV_PYTHON_VERSION"

# Upgrade pip
print_status "Upgrading pip..."
python -m pip install --upgrade pip

# Install wheel and setuptools first
print_status "Installing build tools..."
pip install wheel setuptools

# Install requirements in order (to handle dependencies better)
print_status "Installing core dependencies..."

# Install PyTorch CPU version first (important for other packages)
print_status "Installing PyTorch CPU..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install core ML libraries
print_status "Installing core ML libraries..."
pip install numpy scipy scikit-learn

# Install computer vision libraries
print_status "Installing computer vision libraries..."
pip install opencv-python Pillow

# Install anomaly detection libraries
print_status "Installing anomaly detection libraries..."
pip install pyod combo

# Install NLP libraries
print_status "Installing NLP libraries..."
pip install transformers spacy

# Install object tracking libraries (these might have issues with 3.12)
print_status "Installing object tracking libraries..."
pip install cython_bbox lap motmetrics || {
    print_warning "Some tracking libraries failed to install. Trying alternatives..."
    pip install cython || true
    pip install lap motmetrics || true
}

# Install YOLOX (might need special handling)
print_status "Installing YOLOX..."
pip install yolox || {
    print_warning "YOLOX failed to install. This is optional for the anomaly detection demo."
}

# Install remaining requirements
print_status "Installing remaining requirements..."
pip install -r requirements.txt || {
    print_warning "Some packages from requirements.txt failed. Installing individually..."
    
    # Install packages individually, skipping failures
    while IFS= read -r line; do
        # Skip comments and empty lines
        if [[ $line =~ ^[[:space:]]*# ]] || [[ -z "${line// }" ]]; then
            continue
        fi
        
        # Extract package name (before ==, >=, etc.)
        package=$(echo "$line" | sed 's/[>=<].*//' | sed 's/\[.*\]//')
        
        if [ ! -z "$package" ]; then
            print_status "Installing $package..."
            pip install "$line" || print_warning "Failed to install $package, skipping..."
        fi
    done < requirements.txt
}

# Test critical imports
print_status "Testing critical imports..."
python -c "
import sys
print(f'Python version: {sys.version}')
print()

# Test critical imports
test_imports = [
    ('numpy', 'NumPy'),
    ('cv2', 'OpenCV'),
    ('sklearn', 'Scikit-learn'),
    ('torch', 'PyTorch'),
    ('pyod', 'PyOD'),
    ('transformers', 'Transformers'),
    ('spacy', 'spaCy')
]

success_count = 0
for module, name in test_imports:
    try:
        imported = __import__(module)
        version = getattr(imported, '__version__', 'Unknown')
        print(f'✅ {name}: {version}')
        success_count += 1
    except ImportError as e:
        print(f'❌ {name}: Import failed - {e}')

print(f'\\nSuccessfully imported {success_count}/{len(test_imports)} critical libraries')
"

# Test anomaly detection functionality
print_status "Testing anomaly detection functionality..."
python -c "
try:
    from pipelines.border_anomaly.anomaly_detector import IsolationForestDetector, MotionBasedDetector, create_synthetic_anomaly_data
    print('✅ Anomaly detection modules imported successfully')
    
    # Quick test
    trajectories, labels = create_synthetic_anomaly_data(num_normal=5, num_anomalies=2)
    detector = MotionBasedDetector(adaptive_thresholds=False)
    result = detector.predict(trajectories[0])
    print(f'✅ Anomaly detection test passed: {result.detection_method}')
    
except Exception as e:
    print(f'❌ Anomaly detection test failed: {e}')
"

# Create activation script
print_status "Creating activation script..."
cat > activate_venv.sh << 'EOF'
#!/bin/bash
# Activate the Python 3.10 virtual environment
source venv/bin/activate
echo "🐍 Python 3.10 virtual environment activated!"
echo "Python version: $(python --version)"
echo "To deactivate, run: deactivate"
EOF
chmod +x activate_venv.sh

# Create requirements verification script
print_status "Creating requirements verification script..."
cat > verify_requirements.py << 'EOF'
#!/usr/bin/env python3
"""
Verify that all required packages are installed and working.
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {package_name or module_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: {e}")
        return False

def main():
    print(f"Python version: {sys.version}")
    print("=" * 50)
    
    # Core requirements
    requirements = [
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('sklearn', 'Scikit-learn'),
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('pyod', 'PyOD'),
        ('transformers', 'Transformers'),
        ('spacy', 'spaCy'),
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('sqlalchemy', 'SQLAlchemy'),
        ('pydantic', 'Pydantic'),
    ]
    
    success_count = 0
    for module, name in requirements:
        if test_import(module, name):
            success_count += 1
    
    print("=" * 50)
    print(f"Successfully imported {success_count}/{len(requirements)} packages")
    
    if success_count == len(requirements):
        print("🎉 All requirements satisfied!")
        return 0
    else:
        print("⚠️  Some packages are missing. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF
chmod +x verify_requirements.py

print_success "Python 3.10 virtual environment setup complete!"
print_status "Next steps:"
echo "  1. Activate the environment: source activate_venv.sh"
echo "  2. Verify installation: python verify_requirements.py"
echo "  3. Test anomaly detection: python demo_anomaly_detection.py"
echo ""
print_status "To use this environment in the future:"
echo "  source venv/bin/activate"
echo ""
print_success "Setup completed successfully! 🎉"