let model;
let pictureToProcess;

$(document).ready()
{
    console.log("Ready!");
    loadModel();
}

//https://www.youtube.com/watch?v=rn19pPYrHg0

async function loadModel() {
    console.log("Loading model...");
    //$('.progress-bar').show();
    $('#progressArea').removeClass("hidden");
    model = await tf.loadLayersModel('models/model.json');
    // $('.progress-bar').hide();
    $('#progressArea').addClass("hidden");
    console.log("Model loaded ...");
}

function displaySummary(){
    model.summary();
}

async function predict() {
    let image = $('imagePreview');

    //Convert it to a Tensor3D
    //The img element is already in the required size, no need to resize?

    let tensor = tf.fromPixels(image)
                 .toFloat();

    let normalized = tensorImage.mul(1.0 / 255.0);
    let expanded = normalized.expandDims(0);

    let prediction = await model.predict(expanded).data();
    console.log(prediction);

}

function fileSelected() {
    console.log("Image selected");

    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;

        pictureToProcess = new Image();
        pictureToProcess.src = dataURL;

        $("#imagePreview").on("load", doInference);
        $("#imagePreview").attr("src", dataURL);
        //$("#userMessage").html('<div class="alert alert-info text-center">Click the Predict! button to get a prediction<div>');
    }

    let file = $("#pictureUploader").prop('files')[0];
    reader.readAsDataURL(file);
}


async function doInference() {

    const IMAGE_SIZE = 150;
    console.log("Doing inference");
    $("#userMessage").html("");

    let image = $('#imagePreview').get(0);
    
    let tensorImage = tf.browser.fromPixels(pictureToProcess)
        .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE])
        .toFloat();

    console.log("Normalizing");
    let normalized = tensorImage.mul(1.0 / 255.0);
    let batched = normalized.expandDims(0);

     console.log("Running prediction");
     let prediction = await model.predict(batched).data();

     if (prediction) {
         const catProbability = prediction[0] * 100;
         const dogProbability = prediction[1] * 100;
         console.log(catProbability);
         console.log(dogProbability);
         
         
         if (catProbability > 50.0)
             $("#predictedClassName").html(`I think this is a cat 😸<br/>I'm <span>${catProbability.toFixed(2)}%</span> certain`);
        else
             $("#predictedClassName").html(`I think this is a dog 🐶<br/>I'm <span>${dogProbability.toFixed(2)}%</span> certain`);
         
     }
    
    console.log("Image processing completed :)");
}

