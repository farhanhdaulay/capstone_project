#!/bin/bash
echo "=== Cleaning venv torch (keeping system CUDA torch) ==="
VENV_SITE=~/dms_project/.venv/lib/python3.10/site-packages

# Remove any venv-installed torch
rm -rf $VENV_SITE/torch
rm -rf $VENV_SITE/torch-*.dist-info
rm -rf $VENV_SITE/functorch
rm -rf $VENV_SITE/torchgen
rm -rf $VENV_SITE/torch.libs
rm -rf $VENV_SITE/torchvision
rm -rf $VENV_SITE/torchvision-*.dist-info
rm -rf $VENV_SITE/torchvision.libs
# Re-enable system site packages
sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' \
  ~/dms_project/.venv/pyvenv.cfg

echo "=== Verifying torch ==="
~/dms_project/.venv/bin/python -c "
import torch
print('Torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('From:', torch.__file__)
"
