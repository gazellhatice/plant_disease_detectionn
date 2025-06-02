<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Object Detection Web App</title>
    <style>
        body { text-align: center; font-family: Arial, sans-serif; }
        canvas { border: 2px solid black; }
    </style>
</head>
<body>
    <h1>Upload an Image for Object Detection</h1>
    <input type="file" id="imageUpload" accept="image/*">
    <br><br>
    <canvas id="canvas"></canvas>
    <script>
        document.getElementById("imageUpload").addEventListener("change", async function(event) {
            let file = event.target.files[0];
            if (!file) return;

            let formData = new FormData();
            formData.append("image", file);

            let response = await fetch("http://127.0.0.1:5000/upload", {
                method: "POST",
                body: formData
            });

            let data = await response.json();
            if (data.error) {
                alert(data.error);
                return;
            }

            // Load the image onto the canvas
            let img = new Image();
            img.src = URL.createObjectURL(file);
            img.onload = function() {
                let canvas = document.getElementById("canvas");
                let ctx = canvas.getContext("2d");

                // Set canvas size
                canvas.width = img.width / 2;
                canvas.height = img.height / 2;

                // Draw the uploaded image
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                // Draw bounding boxes
                data.detections.forEach(det => {
                    let [x1, y1, x2, y2] = det.bbox.map(val => val / 2);
                    ctx.strokeStyle = "red";
                    ctx.lineWidth = 3;
                    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                    // Display label and confidence
                    ctx.fillStyle = "red";
                    ctx.font = "18px Arial";
                    ctx.fillText(`${det.label} (${det.confidence}%)`, x1, y1 - 5);
                });
            };
        });
    </script>
</body>
</html>
