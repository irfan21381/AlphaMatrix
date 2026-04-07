# This file bridges the OpenEnv Validator to your actual backend
from app.main import app as main_app

# The validator expects the variable to be named 'app'
app = main_app