#!/usr/bin/env python3
"""
Main runner script for Neural Style Transfer application.
Provides easy commands to start different components.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def run_streamlit():
    """Run Streamlit UI"""
    print("🎨 Starting Streamlit UI...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "src/ui/streamlit_app.py",
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ])

def run_api():
    """Run FastAPI backend"""
    print("🚀 Starting FastAPI backend...")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "src.api.main:app",
        "--host=0.0.0.0",
        "--port=8000",
        "--reload"
    ])

def setup_database():
    """Initialize database with mock data"""
    print("🗄️ Setting up database...")
    subprocess.run([sys.executable, "src/database/models.py"])
    print("✅ Database initialized with sample templates!")

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Requirements installed!")

def run_docker():
    """Run with Docker Compose"""
    print("🐳 Starting with Docker Compose...")
    subprocess.run(["docker-compose", "up", "--build"])

def main():
    parser = argparse.ArgumentParser(description="Neural Style Transfer Application Runner")
    parser.add_argument("command", choices=[
        "ui", "api", "setup", "install", "docker", "all"
    ], help="Command to run")
    
    args = parser.parse_args()
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    if args.command == "install":
        install_requirements()
    elif args.command == "setup":
        setup_database()
    elif args.command == "ui":
        run_streamlit()
    elif args.command == "api":
        run_api()
    elif args.command == "docker":
        run_docker()
    elif args.command == "all":
        print("🚀 Setting up complete environment...")
        install_requirements()
        setup_database()
        print("\n" + "="*50)
        print("✅ Setup complete!")
        print("Run 'python run.py ui' to start the Streamlit interface")
        print("Run 'python run.py api' to start the FastAPI backend")
        print("Run 'python run.py docker' to start with Docker")

if __name__ == "__main__":
    main()
