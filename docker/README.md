# Running BLITZ in a Browser with Docker

You can run BLITZ directly in your web browser using Docker. This provides a full desktop environment where you can use the application just like on a native machine.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your machine.
- [Docker Compose](https://docs.docker.com/compose/install/) (included with Docker Desktop).

## Quick Start

1.  Open a terminal in the **repository root**.
2.  Build the Docker image:
    ```bash
    docker compose -f docker/docker-compose.yml build
    ```
    *(This may take a few minutes as it installs all dependencies)*

3.  Start the container:
    ```bash
    docker compose -f docker/docker-compose.yml up -d
    ```

4.  Open your browser and go to:
    [http://localhost:3000](http://localhost:3000)

5.  You will see a Linux desktop. **Double-click the "BLITZ" icon on the desktop** to start the application.

## Transferring Files

### Option 1: The `data` Folder (Recommended)
The easiest way to load your images is to place them in the `data` folder on your host machine (created automatically at the repository root when the container starts).

1.  Copy your images/videos to the `./data` folder at the repository root.
2.  In BLITZ, navigate to `/data` to open them.

### Option 2: Drag and Drop
You can drag files from your computer directly onto the browser window.
-   Files will be uploaded to the `/config/Downloads` or `/config` directory inside the container.
-   Navigate there in BLITZ to open them.

## Stopping

To stop the container (from the repository root):
```bash
docker compose -f docker/docker-compose.yml down
```

## Troubleshooting

-   **Performance:** If the interface is slow, ensure Docker has access to enough CPU/RAM. BLITZ is optimized for CPU processing.
-   **OpenGL Errors:** If BLITZ fails to start due to graphics issues, check the Docker logs (`docker compose -f docker/docker-compose.yml logs blitz`). The image includes software rendering (llvmpipe) which should work on most systems.
