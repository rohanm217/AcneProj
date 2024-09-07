async function postimage() {
    try {
        const canvasElement = document.getElementById('canvasElement');
        const capturedImage = document.getElementById('capturedImage');
        const capturedImageData = canvasElement.toDataURL('image/png');

        const response = await fetch("http://127.0.0.1:8000/image", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ image_base64: capturedImageData }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const responseData = await response.json();
        const resultImageBase64 = responseData.image_base64;
        const classes = responseData.classes;

        console.log("Classes found:", classes);

        capturedImage.src = `data:image/png;base64,${resultImageBase64}`;
        capturedImage.style.display = 'block';
    } catch (error) {
        console.error('Error processing image:', error);
        resultMessage.textContent = 'Failed to process the image.';
    }
}
document.addEventListener('DOMContentLoaded', () => {
    const picbutton = document.getElementById('picbutton');
    const resultMessage = document.getElementById('resultMessage');
    const photoContainer = document.getElementById('photoContainer');
    const videoElement = document.getElementById('videoElement');
    const captureButton = document.getElementById('captureButton');
    const canvasElement = document.getElementById('canvasElement');
    const capturedImage = document.getElementById('capturedImage');


    picbutton.addEventListener('click', async () => {
        try {

            const stream = await navigator.mediaDevices.getUserMedia({ video: true });

            videoElement.srcObject = stream;
            photoContainer.style.display = 'block';
            resultMessage.textContent = 'Click "Capture" to take a picture.';

            picbutton.style.display = 'none';
        } catch (error) {
            console.error('Error accessing the camera:', error);
            resultMessage.textContent = 'Failed to access the camera.';
        }
    });

    captureButton.addEventListener('click', () => {
        const context = canvasElement.getContext('2d');

        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;

        context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

        capturedImage.src = canvasElement.toDataURL('image/png');

        const stream = videoElement.srcObject;
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());

        videoElement.style.display = 'none';
        captureButton.style.display = 'none';
        document.getElementById("capturedImage").style.display="inline"
    });
});