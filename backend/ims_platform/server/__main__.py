"""Run with: python -m ims_platform.server"""
from .app import create_app

if __name__ == "__main__":
    app = create_app()
    print("IMS Analyzer (real-backend edition) running at http://127.0.0.1:8765/")
    app.run(host="127.0.0.1", port=8765, debug=False)
