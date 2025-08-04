# Drag Coefficient Prediction App

This project is a web application that predicts the drag coefficient (Cd) of a vehicle based on its 3D point cloud data. The application is built with Streamlit and served using Docker.

## Research Paper

The models and methods used in this project are based on the following research paper. The paper provides a detailed explanation of the data processing, model architecture, and experimental results.

**[→ Read the Full Research Paper (PDF)](./Cd_Prediction_Research_Paper.pdf)**

## Quickstart: Running the Application Locally

The fastest way to get the application running is with Docker and Docker Compose.

### Prerequisites

*   [Docker](https://docs.docker.com/get-docker/)

### Instructions

1.  **Clone the repository:**

    ```bash
    git clone git remote add origin git@github.com:Adarsh-Roy/cd_prediction.git
    cd cd_prediction
    ```

2.  **Build and run the container:**

    ```bash
    docker build -t cd-prediction-app . && docker run -p 8501:8501 cd-prediction-app
    ```

3.  **Open the application:**

    Once the container is running, open your web browser and navigate to:
    [http://localhost:8501](http://localhost:8501)

## How to Use the App

1.  **Upload a file:** Use the file uploader to select a point cloud file. The application supports both `.pcd` and `.paddle_tensor` formats.
2.  **View the prediction:** The application will process the file and display the predicted drag coefficient (Cd) value.

## For Development: Model Training

This repository is a monorepo containing both the Streamlit application and the complete code for model training and experimentation.

If you are interested in the details of the data preprocessing, model architecture, or training process, please refer to the detailed documentation in the `training` directory:
The model file present in this repository is obtained by training on DriveAer++ dataset with around 8000 car point clouds.

## Authors

- **Adarsh Roy** - [GitHub](https://github.com/Adarsh-Roy)
- **Utkarsh Singh** - [GitHub](https://github.com/HSRAKTU)
- **Absaar Ali** - [GitHub](https://github.com/Absaar1548)

## Todo
- [ ] Support points cloud uploads in any orientation. The current implementation only handles point clouds in standard orientation, that is, the rear end to the front end of the car is along the positive x axis.
- [ ] Support standard cad model file formats, like .stl.

**[→ Go to Training Documentation](./training/README.md)**
