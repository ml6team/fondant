import os
import sys

# Get the absolute path to the "src" directory
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))

# Append the "src" directory to the Python path
sys.path.append(src_path)
