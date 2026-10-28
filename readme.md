python -m venv venv
venv\Scripts\activate
cd thermal_app
python create_folders.py
pip install -r requirements.txt
python trainner.py
python app.py 