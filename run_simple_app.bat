@echo off
echo Installing dependencies...
pip install streamlit numpy matplotlib

echo Starting TCQ Demonstration App...
start http://localhost:8501
streamlit run simple_app.py

pause
