# Use the linuxserver/webtop image based on Ubuntu 24.04 (Noble) with Openbox (Lightweight Kiosk)
FROM lscr.io/linuxserver/webtop:ubuntu-openbox

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_SYSTEM_PYTHON=1

# Install system dependencies
# - python3-venv for uv
# - libgl1 and libglx-mesa0 for OpenGL (PyQt6/cv2)
# - libglib2.0-0, libsm6, libxext6, libxrender-dev for cv2
# - libxcb-cursor0 for Qt6
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    libgl1 \
    libglx-mesa0 \
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

# Copy the repository content into the image (build context is repo root)
COPY . .

# Create a virtual environment and install dependencies
RUN uv sync --frozen

# Install the startup script
COPY docker/start-blitz.sh /usr/local/bin/start-blitz.sh
RUN chmod +x /usr/local/bin/start-blitz.sh

# Configure Openbox for Kiosk mode
# Copy configuration files (rc.xml, menu.xml, autostart) to system defaults and /defaults for user init
RUN mkdir -p /etc/xdg/openbox /defaults/openbox && \
    cp docker/openbox/* /etc/xdg/openbox/ && \
    cp docker/openbox/* /defaults/openbox/ && \
    chmod +x /etc/xdg/openbox/autostart /defaults/openbox/autostart

# Install the desktop shortcut (optional backup)
COPY docker/BLITZ.desktop /usr/share/applications/BLITZ.desktop
RUN chmod +x /usr/share/applications/BLITZ.desktop

# Ensure permissions for the app directory
RUN chown -R abc:abc /app/blitz

# Copy the desktop shortcut to the user's desktop (via init script)
RUN mkdir -p /etc/cont-init.d && \
    echo '#!/bin/bash\n\
mkdir -p /config/Desktop\n\
cp /usr/share/applications/BLITZ.desktop /config/Desktop/BLITZ.desktop\n\
chown abc:abc /config/Desktop/BLITZ.desktop\n\
chmod +x /config/Desktop/BLITZ.desktop\n\
# Ensure Openbox config is populated if missing\n\
mkdir -p /config/.config/openbox\n\
if [ ! -f /config/.config/openbox/rc.xml ]; then\n\
    cp /defaults/openbox/* /config/.config/openbox/\n\
    chown -R abc:abc /config/.config/openbox\n\
    chmod +x /config/.config/openbox/autostart\n\
fi\n\
' > /etc/cont-init.d/99-blitz-config && \
    chmod +x /etc/cont-init.d/99-blitz-config
