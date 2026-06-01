"""
Environment checker for NeuroScan project.
Detects Python version and whether required packages are installed,
and prints step-by-step commands to create a compatible virtualenv and install dependencies.
"""
import sys
import subprocess
import importlib

REQUIRED_PACKAGES = [
    'tensorflow',
    'scikit_learn',
    'numpy',
    'matplotlib',
    'seaborn',
    'PIL'
]


def check_python_version():
    major = sys.version_info.major
    minor = sys.version_info.minor
    print(f"Python version: {major}.{minor}")
    supported_versions = [(3, 11), (3, 12), (3, 14)]
    if (major, minor) in supported_versions:
        print("Python version is supported for TensorFlow if the correct wheel is installed.\n")
        return True
    else:
        print("WARNING: TensorFlow may not support this Python version.\nPlease install Python 3.11, 3.12, or 3.14 with a compatible TensorFlow wheel.")
        return False


def check_packages():
    print("Checking installed packages...")
    missing = []
    for pkg in ['tensorflow', 'scikit_learn', 'numpy', 'matplotlib', 'seaborn', 'PIL']:
        try:
            if pkg == 'scikit_learn':
                import sklearn  # noqa: F401
            elif pkg == 'PIL':
                from PIL import Image  # noqa: F401
            else:
                importlib.import_module(pkg)
            print(f"  - {pkg}: OK")
        except Exception:
            print(f"  - {pkg}: MISSING")
            missing.append(pkg)
    return missing


def print_fix_instructions():
    print('\nRecommended steps (Windows):\n')
    print('1) Install Python 3.11, 3.12, or 3.14 if not installed, or use an existing Python 3.14 environment with TensorFlow already available:')
    print('   - Download from https://www.python.org/downloads/release/python-311/')
    print('   - Or use winget:')
    print('     winget install --id Python.Python.3.11')
    print('\n2) Recreate virtual environment in project root:')
    print('   cd C:\\Users\\AARJU\\Documents\\AI_PRO')
    print('   Remove-Item -Recurse -Force venv')
    print('   py -3.11 -m venv venv')
    print('   .\\venv\\Scripts\\Activate.ps1')
    print('   python -m pip install --upgrade pip')
    print('   python -m pip install -r requirements.txt')
    print('   # If TensorFlow cannot be installed with this interpreter, use the demo-only fallback:')
    print('   # python -m pip install -r requirements_no_tf.txt')
    print('\n3) Run the project:')
    print('   cd NeuroScan')
    print('   python app.py')


if __name__ == '__main__':
    ok = check_python_version()
    missing = check_packages()
    if not ok or missing:
        print('\nEnvironment is not fully ready.')
        print_fix_instructions()
    else:
        print('\nEnvironment looks OK. You can run the project with:')
        print('  cd NeuroScan')
        print('  python app.py')
