#!/bin/bash
#
# Checklist and validation script for GeoLexels + SuiT integration
# Run this to verify everything is set up correctly
#

set +e  # Don't exit on errors

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
SUIT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

PASS=0
FAIL=0

check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((PASS++))
    else
        echo -e "${RED}✗${NC} $1"
        ((FAIL++))
    fi
}

echo -e "${BLUE}========== GeoLexels + SuiT Integration Checklist ==========${NC}\n"

# Check 1: Python environment
echo "1. Python & Dependencies:"
python3 --version > /dev/null 2>&1
check "Python 3 available"

python3 -c "import numpy" 2>/dev/null
check "NumPy installed"

python3 -c "import torch" 2>/dev/null
check "PyTorch installed"

python3 -c "import cv2" 2>/dev/null
check "OpenCV (cv2) installed"

python3 -c "import sklearn" 2>/dev/null
check "scikit-learn installed"

python3 -c "import torch; src=torch.randn(2,4); idx=torch.tensor([[0,1,1,2],[0,0,2,2]]); out=torch.zeros(2,3).scatter_reduce(1, idx, src, reduce='sum'); print(out.shape)" 2>/dev/null
check "PyTorch scatter_reduce available"

# Check 2: Project structure
echo -e "\n2. Project Structure:"

[ -f "$PROJECT_ROOT/pointcloud/CMakeLists.txt" ]
check "Pointcloud source code"

[ -f "$PROJECT_ROOT/datasets/SUNRGBD/.DS_Store" ] || [ -d "$PROJECT_ROOT/datasets/SUNRGBD/kv2" ]
check "SUNRGBD dataset exists"

[ -d "$SUIT_DIR" ] && [ -f "$SUIT_DIR/main.py" ]
check "SuiT code available"

# Check 3: Executables
echo -e "\n3. Executables:"

[ -f "$PROJECT_ROOT/pointcloud/build/fast_cloud" ] && [ -x "$PROJECT_ROOT/pointcloud/build/fast_cloud" ]
check "fast_cloud built"

# Check 4: Scripts
echo -e "\n4. Integration Scripts:"

[ -f "$SUIT_DIR/precompute_geolexels.py" ] && [ -x "$SUIT_DIR/precompute_geolexels.py" ]
check "precompute_geolexels.py"

[ -f "$SUIT_DIR/run_geolexels_preprocessing.sh" ] && [ -x "$SUIT_DIR/run_geolexels_preprocessing.sh" ]
check "run_geolexels_preprocessing.sh"

[ -f "$SUIT_DIR/run_suit_training.sh" ] && [ -x "$SUIT_DIR/run_suit_training.sh" ]
check "run_suit_training.sh"

[ -f "$SUIT_DIR/run_complete_workflow.sh" ] && [ -x "$SUIT_DIR/run_complete_workflow.sh" ]
check "run_complete_workflow.sh"

# Check 5: Documentation
echo -e "\n5. Documentation:"

[ -f "$SUIT_DIR/GEOLEXELS_INTEGRATION.md" ]
check "GEOLEXELS_INTEGRATION.md"

[ -f "$SUIT_DIR/README_GEOLEXELS.md" ]
check "README_GEOLEXELS.md"

# Check 6: Dataset verification
echo -e "\n6. Dataset Integrity:"

if [ -d "$PROJECT_ROOT/datasets/SUNRGBD" ]; then
    RGB_COUNT=$(find "$PROJECT_ROOT/datasets/SUNRGBD" -name "*.jpg" 2>/dev/null | wc -l)
    [ $RGB_COUNT -gt 0 ]
    check "Found $RGB_COUNT RGB images"
    
    DEPTH_COUNT=$(find "$PROJECT_ROOT/datasets/SUNRGBD" -name "depth/*.png" 2>/dev/null | wc -l)
    [ $DEPTH_COUNT -gt 0 ]
    check "Found $DEPTH_COUNT depth images"
else
    echo -e "${YELLOW}⊘${NC} SUNRGBD dataset not found (optional for setup validation)"
fi

# Check 7: Modified code
echo -e "\n7. Code Integration:"

grep -q "SUNRGBDGeolexelsDataset" "$SUIT_DIR/datasets.py"
check "SUNRGBDGeolexelsDataset in datasets.py"

grep -q "data-set SUNRGBD" "$SUIT_DIR/run_suit_training.sh"
check "SUNRGBD support in training script"

# Check 8: Storage
echo -e "\n8. Storage Status:"

CACHE_DIR="$PROJECT_ROOT/datasets/SUNRGBD/.geolexels_cache"
if [ -d "$CACHE_DIR" ]; then
    CACHE_COUNT=$(find "$CACHE_DIR" -name "*.npy" 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} Found $CACHE_COUNT cached GeoLexels files"
    ((PASS++))
else
    echo -e "${YELLOW}⊘${NC} Cache directory doesn't exist yet (run preprocessing first)"
fi

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}"
echo -e "${BLUE}========================================${NC}\n"

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo -e "\nYou're ready to start. Try:"
    echo -e "  ${YELLOW}cd $SUIT_DIR${NC}"
    echo -e "  ${YELLOW}bash run_complete_workflow.sh --max-images 10${NC}"
    exit 0
else
    echo -e "${RED}✗ Some checks failed. Please fix the issues above.${NC}"
    echo -e "\nCommon fixes:"
    echo "  - Install missing packages: pip install numpy torch opencv-python scikit-learn"
    echo "  - Build fast_cloud: cd pointcloud && mkdir build && cd build && cmake .. && make"
    echo "  - Check documentation: cat $SUIT_DIR/GEOLEXELS_INTEGRATION.md"
    exit 1
fi
