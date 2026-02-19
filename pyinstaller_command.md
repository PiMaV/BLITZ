uv sync

uv add --dev pyinstaller

uv venv   # oder uv run -- aktiviert venv nicht explizit

### requirements pruefen
uv pip list

uv run pyinstaller --onefile --windowed --clean --name BLITZ --icon=./resources/icon/blitz.ico blitz_main.py