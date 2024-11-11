import { useState } from "react";
import axios from "axios";

function App() {
  const [image, setImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [model, setModel] = useState("yolo"); // State for selected model

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
    setProcessedImage(null);
    setError(null);
  };

  const handleModelChange = (e) => {
    setModel(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", image);

    // Select the endpoint based on chosen model
    const endpoint =
      model === "yolo"
        ? "http://localhost:8000/inference/yolo"
        : model === "mask-rcnn"
        ? "http://localhost:8000/inference/maskrcnn"
        : "http://localhost:8000/inference/yolos"; // Endpoint for YOLOS model

    try {
      const response = await axios.post(endpoint, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        responseType: "blob",
      });

      const imageURL = URL.createObjectURL(new Blob([response.data]));
      setProcessedImage(imageURL);
      setError(null);
    } catch (error) {
      console.error("Error:", error);
      setError("Failed to process the image. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-r from-gray-100 to-gray-200 flex flex-col items-center justify-center p-4">
      <div className="p-8 bg-white rounded-xl shadow-xl w-full max-w-lg">
        <h1 className="text-4xl font-bold text-center mb-8 text-gray-800">
          Shelf Empty Space Detection
        </h1>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Model selection dropdown */}
          <select
            value={model}
            onChange={handleModelChange}
            className="w-full py-2 px-4 bg-gray-100 rounded-lg shadow-md text-gray-700 focus:outline-none"
          >
            <option value="yolo">YOLO Model</option>
            <option value="mask-rcnn">Mask R-CNN Model</option>
            <option value="yolos">YOLOS Model</option>{" "}
            {/* New option for YOLOS model */}
          </select>

          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4
                      file:rounded-full file:border-0 file:text-sm file:font-semibold
                      file:bg-blue-100 file:text-blue-700 hover:file:bg-blue-200
                      transition duration-200 ease-in-out"
          />

          <button
            type="submit"
            className="w-full py-2 px-4 bg-blue-500 text-white rounded-lg shadow-md 
                       hover:bg-blue-600 hover:scale-105 transition-all duration-300 ease-in-out
                       focus:outline-none focus:ring-4 focus:ring-blue-300"
          >
            {loading ? "Processing..." : "Upload and Detect"}
          </button>
        </form>

        {error && <div className="mt-4 text-red-500 text-center">{error}</div>}

        {image && (
          <div className="mt-6">
            <h2 className="text-lg font-semibold mb-2">Uploaded Image:</h2>
            <img
              src={URL.createObjectURL(image)}
              alt="Uploaded preview"
              className="rounded-lg shadow-md w-full"
            />
          </div>
        )}

        {loading && (
          <div className="mt-6 text-center">
            <div className="loader border-t-blue-500 border-4 rounded-full w-12 h-12 animate-spin mx-auto"></div>
            <p className="mt-4 text-gray-600">Detecting empty spaces...</p>
          </div>
        )}

        {processedImage && (
          <div className="mt-6">
            <h2 className="text-lg font-semibold mb-2">Processed Image:</h2>
            <img
              src={processedImage}
              alt="Processed result"
              className="rounded-lg shadow-md w-full"
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
