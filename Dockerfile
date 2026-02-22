# Use the linuxserver/webtop image based on Ubuntu 24.04 (Noble) which includes Python 3.12
FROM lscr.io/linuxserver/webtop:ubuntu-xfce

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_SYSTEM_PYTHON=1

# Install system dependencies
# - python3-venv for uv
# - libgl1-mesa-glx for OpenGL (PyQt6/cv2)
# - libglib2.0-0, libsm6, libxext6, libxrender-dev for cv2
# - libxcb-cursor0 for Qt6
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
RUN pip3 install uv --break-system-packages

# Create directory for the application
WORKDIR /app/blitz

# Copy the repository content into the image
COPY . .

# Create a virtual environment and install dependencies
# We use --system to install into the system python or --python to specify version if needed
# But here we want to use a venv in the project directory for isolation
RUN uv sync --frozen

# Install the startup script
COPY start-blitz.sh /usr/local/bin/start-blitz.sh
RUN chmod +x /usr/local/bin/start-blitz.sh

# Install the desktop shortcut
COPY BLITZ.desktop /usr/share/applications/BLITZ.desktop
RUN chmod +x /usr/share/applications/BLITZ.desktop

# Ensure permissions for the app directory
RUN chown -R abc:abc /app/blitz

# Copy the desktop shortcut to the user's desktop (needs to be done at runtime via init, or baked in)
# Webtop populates /config/Desktop from /defaults/Desktop if it exists? No, it doesn't.
# But we can add a custom service or just rely on the menu entry.
# However, for "noob proven", a desktop icon is best.
# We can create a script in /etc/cont-init.d/99-blitz-shortcut
RUN mkdir -p /etc/cont-init.d && \
    echo '#!/bin/bash\n\
mkdir -p /config/Desktop\n\
cp /usr/share/applications/BLITZ.desktop /config/Desktop/BLITZ.desktop\n\
chown abc:abc /config/Desktop/BLITZ.desktop\n\
chmod +x /config/Desktop/BLITZ.desktop\n\
' > /etc/cont-init.d/99-blitz-shortcut && \
    chmod +x /etc/cont-init.d/99-blitz-shortcut
