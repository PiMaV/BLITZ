poetry install

poetry add --group dev pyinstaller

poetry env info

poetry shell
### sind die requirements installiert und nicht
poetry run pip list

# Build using the spec file
poetry run pyinstaller blitz.spec
