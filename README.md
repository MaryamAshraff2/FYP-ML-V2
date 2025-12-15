# FYP-ML-V2: Person Re-identification System

## Project Description

This project implements a Person Re-identification (ReID) system using machine learning techniques, with Qdrant as the vector database for efficient similarity search. The system aims to identify the same person across different camera views, which is crucial for various applications like surveillance, retail analytics, and security.

## Features

*   **Person Re-identification:** Identifies individuals across multiple camera feeds.
*   **Deep Learning Models:** Utilizes state-of-the-art deep learning models (e.g., MobileNet/OSNet) for robust feature extraction.
*   **Vector Database (Qdrant):** Employs Qdrant for high-performance storage and retrieval of person embeddings, enabling fast similarity searches.
*   **Scalable Architecture:** Designed for scalability to handle large volumes of video data and person identities.
*   **Clustering:** Includes clustering functionalities to group similar person embeddings.

## Installation

To set up and run this project locally, follow these steps:

### Prerequisites

*   Python 3.8+
*   pip (Python package installer)

### Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/MaryamAshraff2/FYP-ML-V2.git
    cd FYP-ML-V2
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The project's configuration, especially for Qdrant, is managed via `config/qdrant.yaml`.

### Qdrant Configuration

The `config/qdrant.yaml` file contains settings for connecting to your Qdrant instance, collection details, upload parameters, and search/clustering configurations.

```yaml
# config/qdrant.yaml
qdrant:
  connection:
    url: "YOUR_QDRANT_INSTANCE_URL"  # e.g., "https://your-qdrant-instance.qdrant.tech:6333"
    api_key: "YOUR_QDRANT_API_KEY"  # Set if using Qdrant Cloud
    timeout: 60.0
    prefer_grpc: false
  collection:
    name: "person_reid_embeddings"
    vector_size: 256 # Or 512 if using OSNet
    distance: "Cosine"
    on_disk_payload: false
  # ... other Qdrant settings ...
```

**Important:** Replace `"YOUR_QDRANT_INSTANCE_URL"` and `"YOUR_QDRANT_API_KEY"` with your actual Qdrant Cloud instance URL and API key.

## Usage

This section will describe how to use your Person ReID system.

### Running the Pipeline

The core logic of the system is orchestrated through various pipelines. For example, `core/pipeline/main_pipeline.py` and `core/pipeline/video_pipeline.py` are likely entry points for processing.

Further details on running specific components or the entire system will be added here.

### Example Usage (Placeholder)

```python
# Example of how to use the system
# from core.pipeline.main_pipeline import run_reid_pipeline
#
# video_path = "path/to/your/video.mp4"
# results = run_reid_pipeline(video_path)
# print(results)
```

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

[Specify your project's license here, e.g., MIT, Apache 2.0, etc.]

---

