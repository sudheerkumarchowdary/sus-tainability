# detection.py
from inference_sdk import InferenceHTTPClient
import sys
import json 

# Initialize the Inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="1qZaH267rMRrXuOYBvTt"
)

def run_inference(image_path):
    try:
        # Perform inference on the given image path
        result = CLIENT.infer(image_path, model_id="yolo-waste-detection/1")

        # Extract detected objects from the predictions
        objects = [pred['class'] for pred in result['predictions']]

        return objects

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    # Get the image path from the command line argument
    image_path = sys.argv[1]

    # Run inference and return the result
    objects_detected = run_inference(image_path)

    # Print the result as JSON for the Streamlit app to consume
    print(json.dumps(objects_detected))
