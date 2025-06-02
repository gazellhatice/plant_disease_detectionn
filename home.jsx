const { useState, useRef } = require("react");
import { Camera } from "react-camera-pro";
import { Button } from "@/components/ui/button";

export default function PlantDiseaseDetection() {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const camera = useRef(null);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("image", file);
    setImage(URL.createObjectURL(file));
    
    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    setPrediction(data);
  };

  const handleCapture = async () => {
    if (camera.current) {
      const imageSrc = camera.current.takePhoto();
      setImage(imageSrc);
      const blob = await fetch(imageSrc).then((res) => res.blob());
      const formData = new FormData();
      formData.append("image", blob, "captured.jpg");
      
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setPrediction(data);
    }
  };

  return (
    <div className="flex flex-col items-center p-6">
      <h1 className="text-2xl font-bold mb-4">Plant Disease Detection</h1>
      <input type="file" accept="image/*" onChange={handleUpload} className="mb-4" />
      <Camera ref={camera} aspectRatio={16 / 9} className="w-full h-64 border mb-4" />
      <Button onClick={handleCapture}>Capture Image</Button>
      {image && <img src={image} alt="Preview" className="mt-4 w-64 h-auto border" />}
      {prediction && (
        <div className="mt-4 p-4 border rounded bg-gray-100">
          <h2 className="text-lg font-semibold">Prediction:</h2>
          <p>Disease: {prediction.disease}</p>
          <p>Accuracy: {prediction.confidence}%</p>
        </div>
      )}
    </div>
  );
}
