# Traffic AI


## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

The Traffic AI system provides the following features:

1. **Speed Estimation**: Estimate the speed of vehicles using recorded video footage.
2. **Vehicle Counting**: Count the number of vehicles passing through a specified region.
3. **Automatic Number Plate Recognition (ANPR)**: Detect and read license plates from video footage.
4. **Traffic Rate Calculation**: Calculate the traffic rate based on the frame count and duration.
5. **Visual Output**: Save and review processed video with annotations for speed estimation, vehicle counting, and ANPR.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7+
- Streamlit
- Ultralytics YOLO (for object detection and tracking)
- OpenCV
- NumPy
- Pandas
- SciPy
- EasyOCR (for license plate recognition)
- SORT (Simple Online and Realtime Tracking)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/AtharshKrishnamoorthy/Traffic-AI
    cd Traffic-AI
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Traffic AI system:

1. Navigate to the project directory.
2. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

3. Open your web browser and go to `http://localhost:8501` (or the address provided in the terminal).

### Video Processing

1. Use the sidebar to upload a video file.
2. Select the task you want to perform (Speed Estimation, Vehicle Counting, ANPR).
3. The application will process the video and save the output with annotations.

### Traffic Rate Calculation

1. The system automatically calculates the traffic rate based on the frame count and duration of the video.

## Configuration

Ensure you have the required environment setup for running the app. Modify `app.py` and `config.py` as needed for additional configurations.

## Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch-name`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to the original branch: `git push origin feature-branch-name`.
5. Create a pull request.

Please update tests as appropriate and adhere to the project's coding standards.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact

If you have any questions, feel free to reach out:

- Project Maintainer: Atharsh K
- Email: atharshkrishnamoorthy@gmail.com
- Project Link: [GitHub Repository](https://github.com/AtharshKrishnamoorthy/Traffic-AI)
