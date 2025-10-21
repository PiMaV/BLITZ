poetry install

 poetry add --group dev pyinstaller

 poetry env info

poetry shell   
### sind die requirements installiert und nicht 
poetry run pip list

poetry run pyinstaller --onefile --windowed --clean --name BLITZ --icon=./resources/icon/blitz.ico blitz_main.py