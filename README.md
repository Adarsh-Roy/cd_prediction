# Drag Coefficient Prediction App

This project is a web application that predicts the drag coefficient (Cd) of a vehicle based on its 3D point cloud data. The application is built with Streamlit and served using Docker.

![Streamlit App Screenshot](https://place-hold.it/800x400?text=App+Screenshot+Here) 
*Note: You can replace the placeholder URL with a real screenshot of your running application.*

## Quickstart: Running the Application

The fastest way to get the application running is with Docker and Docker Compose.

### Prerequisites

*   [Docker](https://docs.docker.com/get-docker/)
*   [Docker Compose](https://docs.docker.com/compose/install/)

### Instructions

1.  **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd cd_prediction
    ```

2.  **Build and run the container:**

    ```bash
    docker-compose up --build
    ```

3.  **Open the application:**

    Once the container is running, open your web browser and navigate to:
    [http://localhost:8501](http://localhost:8501)

## How to Use the App

1.  **Upload a file:** Use the file uploader to select a point cloud file. The application supports both `.pcd` and `.paddle_tensor` formats.
2.  **View the prediction:** The application will process the file and display the predicted drag coefficient (Cd) value.

## For Developers: Model Training

This repository is a monorepo containing both the Streamlit application and the complete code for model training and experimentation.

If you are interested in the details of the data preprocessing, model architecture, or training process, please refer to the detailed documentation in the `training` directory:

**[→ Go to Training Documentation](./training/README.md)**

## Technology Stack

*   **Application:** Streamlit
*   **Machine Learning:** PyTorch, PaddlePaddle, Scikit-learn, PyTorch Geometric
*   **Deployment:** Docker, Docker Compose
*   **Data Handling:** NumPy, PyPCD
