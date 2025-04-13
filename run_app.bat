@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo Starting TCQ Demonstration App...
start http://localhost:8501
streamlit run app.py

pause
