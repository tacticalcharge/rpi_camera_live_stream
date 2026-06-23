set -e
echo "Creating python Venv..."
python3 -m venv venv

source venv/bin/activate

if [ -f "requirements.txt" ]; then
	echo "installing requirements..."]
	pip install --upgrade pip
	pip install -r requirements.txt
else
	echo "requirements.txt not found, skipping..."
fi

echo "Setup complete!"

echo "Running App.py..."
python3 app.py