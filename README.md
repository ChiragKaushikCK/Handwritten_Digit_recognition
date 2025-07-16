# Handwritten_Digit_recognition

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras" />
  <img src="https://img.shields.io/badge/Matplotlib-EE6633?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
</div>

<h1 align="center" style="color: #4CAF50; font-family: 'Segoe UI', sans-serif; font-size: 3.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">
  <br>
  <a href="https://github.com/your-username/handwritten-digit-recognition-cnn">
    <img src="https://placehold.co/600x200/4CAF50/FFFFFF?text=Handwritten+Digit+CNN" alt="Handwritten Digit Recognition Banner" style="border-radius: 15px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
  </a>
  <br>
  ✍️ Handwritten Digit Recognition with CNN 🔢
  <br>
</h1>

<p align="center" style="font-size: 1.2em; color: #555; max-width: 800px; margin: 0 auto; line-height: 1.6;">
  Unlock the power of deep learning for image classification with this **Handwritten Digit Recognition** project! This Jupyter Notebook demonstrates how to build, train, and evaluate a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to accurately classify handwritten digits from the famous MNIST dataset. It covers data preparation, CNN architecture design, model training, and making predictions on new images. Perfect for understanding the fundamentals of CNNs for computer vision tasks! 🚀
</p>

<br>

<details style="background-color: #E8F5E9; border-left: 5px solid #4CAF50; padding: 15px; border-radius: 8px; margin: 20px auto; max-width: 700px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
  <summary style="font-size: 1.3em; font-weight: bold; color: #333; cursor: pointer;">Table of Contents</summary>
  <ol style="list-style-type: decimal; padding-left: 25px; line-height: 1.8;">
    <li><a href="#about-the-project" style="color: #4CAF50; text-decoration: none;">📚 About The Project</a></li>
    <li><a href="#dataset" style="color: #4CAF50; text-decoration: none;">🔢 Dataset</a></li>
    <li><a href="#model-architecture" style="color: #4CAF50; text-decoration: none;">🏗️ Model Architecture</a></li>
    <li><a href="#features" style="color: #4CAF50; text-decoration: none;">🎯 Features</a></li>
    <li><a href="#prerequisites" style="color: #4CAF50; text-decoration: none;">🛠️ Prerequisites</a></li>
    <li><a href="#how-to-run" style="color: #4CAF50; text-decoration: none;">📋 How to Run</a></li>
    <li><a href="#example-output" style="color: #4CAF50; text-decoration: none;">📈 Example Output</a></li>
    <li><a href="#code-breakdown" style="color: #4CAF50; text-decoration: none;">🧠 Code Breakdown</a></li>
    <li><a href="#customization-ideas" style="color: #4CAF50; text-decoration: none;">🌈 Customization Ideas</a></li>
    <li><a href="#contribute" style="color: #4CAF50; text-decoration: none;">🤝 Contribute</a></li>
  </ol>
</details>

---

<h2 id="about-the-project" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #8BC34A; padding-bottom: 10px;">
  📚 About The Project
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  This project focuses on building and training a **Convolutional Neural Network (CNN)** for the task of handwritten digit recognition. [cite: uploaded:Handwritten_Digit_Recognition.ipynb] CNNs are particularly well-suited for image-based tasks due to their ability to automatically learn hierarchical features from raw pixel data. This notebook provides a practical guide to:
</p>
<ul style="list-style-type: none; padding: 0; font-size: 1.1em; color: #444;">
  <li style="margin-bottom: 10px; background-color: #F1F8E9; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #4CAF50;">Data Loading & Preprocessing:</strong> Loading the MNIST dataset, reshaping images for CNN input, and normalizing pixel values.
  </li>
  <li style="margin-bottom: 10px; background-color: #F1F8E9; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #4CAF50;">CNN Model Definition:</strong> Constructing a sequential CNN model with `Conv2D`, `MaxPool2D`, `Flatten`, and `Dense` layers using Keras.
  </li>
  <li style="margin-bottom: 10px; background-color: #F1F8E9; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #4CAF50;">Model Compilation & Training:</strong> Configuring the optimizer, loss function, and metrics, then training the model on the dataset.
  </li>
  <li style="margin-bottom: 10px; background-color: #F1F8E9; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #4CAF50;">Prediction on New Images:</strong> Demonstrating how to load an external image, preprocess it, and use the trained model to predict the digit.
  </li>
</ul>

---

<h2 id="dataset" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #8BC34A; padding-bottom: 10px;">
  🔢 Dataset
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  This project exclusively uses the <strong style="color: #4CAF50;">MNIST (Modified National Institute of Standards and Technology)</strong> dataset. [cite: uploaded:Handwritten_Digit_Recognition.ipynb] It is a classic dataset in machine learning and computer vision, consisting of:
</p>
<ul style="list-style-type: disc; padding-left: 20px; font-size: 1.1em; color: #444;">
  <li><strong style="color: #4CAF50;">Training Set:</strong> 60,000 examples of handwritten digits.</li>
  <li><strong style="color: #4CAF50;">Test Set:</strong> 10,000 examples for evaluating model performance.</li>
</ul>
<p style="font-size: 1.1em; color: #444; line-height: 1.6; margin-top: 10px;">
  Each example is a 28x28 pixel grayscale image, associated with a label from 0 to 9, representing the digit depicted. The dataset is directly available through `tensorflow.keras.datasets.mnist.load_data()`, making it easy to access and use.
</p>

---

<h2 id="model-architecture" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #8BC34A; padding-bottom: 10px;">
  🏗️ Model Architecture
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  The Convolutional Neural Network (CNN) designed for this task is a sequential model. [cite: uploaded:Handwritten_Digit_Recognition.ipynb] Its architecture is structured to effectively extract features from image data:
</p>
<div style="background-color: #E0F2F7; border: 1px solid #2196F3; padding: 15px; border-radius: 8px; margin: 20px auto; max-width: 600px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
  <h3 style="color: #4CAF50; font-size: 1.8em; margin-top: 0;">Layers:</h3>
  <ul style="list-style-type: none; padding: 0; font-size: 1.1em; color: #444;">
    <li style="margin-bottom: 10px;"><strong style="color: #4CAF50;">Convolutional Layer (`Conv2D`):</strong> Applies filters to extract features (e.g., edges, textures). Uses `relu` activation.</li>
    <li style="margin-bottom: 10px;"><strong style="color: #4CAF50;">Pooling Layer (`MaxPool2D`):</strong> Reduces the spatial dimensions of the feature maps, helping to make the model more robust to small shifts in input.</li>
    <li style="margin-bottom: 10px;"><strong style="color: #4CAF50;">Flatten Layer:</strong> Converts the 2D feature maps into a 1D vector to be fed into dense layers.</li>
    <li style="margin-bottom: 10px;"><strong style="color: #4CAF50;">Dense Hidden Layer:</strong> A fully connected layer for learning complex patterns from the flattened features. Uses `relu` activation.</li>
    <li style="margin-bottom: 10px;"><strong style="color: #4CAF50;">Output Layer (`Dense`):</strong> The final fully connected layer with `softmax` activation, outputting probabilities for each of the 10 digit classes (0-9).</li>
  </ul>
  <h3 style="color: #4CAF50; font-size: 1.8em; margin-top: 20px;">Compilation:</h3>
  <ul style="list-style-type: none; padding: 0; font-size: 1.1em; color: #444;">
    <li style="margin-bottom: 10px;"><strong style="color: #4CAF50;">Optimizer:</strong> Adam (Adaptive Moment Estimation) - an efficient stochastic optimization algorithm.</li>
    <li style="margin-bottom: 10px;"><strong style="color: #4CAF50;">Loss Function:</strong> Sparse Categorical Crossentropy - suitable for integer-encoded labels in multi-class classification.</li>
    <li style="margin-bottom: 10px;"><strong style="color: #4CAF50;">Metrics:</strong> Accuracy - measures the proportion of correctly classified images.</li>
  </ul>
</div>

---

<h2 id="features" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #8BC34A; padding-bottom: 10px;">
  🎯 Features
</h2>
<ul style="list-style-type: none; padding: 0; font-size: 1.1em; color: #444;">
  <li style="margin-bottom: 15px; background-color: #DCEDC8; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #689F38;">🚀 End-to-End Recognition:</strong> Covers data loading, CNN model building, training, and prediction. [cite: uploaded:Handwritten_Digit_Recognition.ipynb]
  </li>
  <li style="margin-bottom: 15px; background-color: #DCEDC8; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #689F38;">🔍 Image Preprocessing:</strong> Demonstrates necessary steps like reshaping and normalization for image data. [cite: uploaded:Handwritten_Digit_Recognition.ipynb]
  </li>
  <li style="margin-bottom: 15px; background-color: #DCEDC8; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #689F38;">📈 Epoch Experimentation:</strong> Includes a section to demonstrate the impact of increasing the number of training epochs on model performance. [cite: uploaded:Handwritten_Digit_Recognition.ipynb]
  </li>
  <li style="margin-bottom: 15px; background-color: #DCEDC8; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #689F38;">🖼️ External Image Prediction:</strong> Shows how to load and predict on a custom image file, making the model practical. [cite: uploaded:Handwritten_Digit_Recognition.ipynb]
  </li>
</ul>

---

<h2 id="prerequisites" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #8BC34A; padding-bottom: 10px;">
  🛠️ Prerequisites
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  To run this project, ensure you have the following installed:
</p>
<ul style="list-style-type: disc; padding-left: 20px; font-size: 1.1em; color: #444;">
  <li><strong style="color: #4CAF50;">Python 3.x</strong></li>
  <li><strong style="color: #4CAF50;">Jupyter Notebook</strong> (or JupyterLab, Google Colab)</li>
  <li>Required Libraries:
    <pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code>pip install tensorflow numpy matplotlib</code></pre>
    (Note: `keras` is part of `tensorflow` in newer versions)
  </li>
</ul>

---

<h2 id="how-to-run" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #8BC34A; padding-bottom: 10px;">
  📋 How to Run
</h2>
<ol style="list-style-type: decimal; padding-left: 20px; font-size: 1.1em; color: #444; line-height: 1.8;">
  <li style="margin-bottom: 10px;">
    <strong style="color: #4CAF50;">Download the Notebook:</strong>
    <p style="margin-top: 5px;">Download <code>Handwritten_Digit_Recognition.ipynb</code> from this repository.</p>
    <p style="margin-top: 5px;">Alternatively, open it directly in <a href="https://colab.research.google.com/" style="color: #4CAF50; text-decoration: none;">Google Colab</a> for a zero-setup experience.</p>
  </li>
  <li style="margin-bottom: 10px;">
    <strong style="color: #4CAF50;">Prepare External Image (Optional):</strong>
    <p style="margin-top: 5px;">If you want to test with your own handwritten digit image, save it (e.g., as `download.png` or `Sample_Image.jpg`) in the same directory as the notebook, or update the `image_path` variable in the notebook accordingly.</p>
  </li>
  <li style="margin-bottom: 10px;">
    <strong style="color: #4CAF50;">Install Dependencies:</strong>
    <pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code>pip install tensorflow numpy matplotlib</code></pre>
  </li>
  <li style="margin-bottom: 10px;">
    <strong style="color: #4CAF50;">Run the Notebook:</strong>
    <p style="margin-top: 5px;">Open <code>Handwritten_Digit_Recognition.ipynb</code> in Jupyter or Colab.</p>
    <p style="margin-top: 5px;">Execute each cell sequentially to train the CNN and make predictions!</p>
  </li>
</ol>

---

<h2 id="example-output" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #8BC34A; padding-bottom: 10px;">
  📈 Example Output
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  The notebook will display training progress and prediction results.
</p>
<h3 style="color: #4CAF50; font-size: 1.8em; margin-top: 25px;">Model Training Output (Epochs):</h3>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #CE9178;">Epoch 1/10</span>
<span style="color: #B5CEA8;">1875/1875</span> <span style="color: #9CDCFE;">━━━━━━━━━━━━━━━━━━━━</span> <span style="color: #B5CEA8;">36s</span> <span style="color: #B5CEA8;">18ms/step - accuracy: 0.9094 - loss: 0.3103</span>
<span style="color: #CE9178;">Epoch 2/10</span>
<span style="color: #B5CEA8;">1875/1875</span> <span style="color: #9CDCFE;">━━━━━━━━━━━━━━━━━━━━</span> <span style="color: #B5CEA8;">38s</span> <span style="color: #B5CEA8;">17ms/step - accuracy: 0.9832 - loss: 0.0548</span>
<span style="color: #CE9178;">...</span>
<span style="color: #CE9178;">Epoch 10/10</span>
<span style="color: #B5CEA8;">1875/1875</span> <span style="color: #9CDCFE;">━━━━━━━━━━━━━━━━━━━━</span> <span style="color: #B5CEA8;">33s</span> <span style="color: #B5CEA8;">18ms/step - accuracy: 0.9981 - loss: 0.0053</span></code></pre>

<h3 style="color: #4CAF50; font-size: 1.8em; margin-top: 25px;">Sample Prediction Output:</h3>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #B5CEA8;">1/1</span> <span style="color: #9CDCFE;">━━━━━━━━━━━━━━━━━━━━</span> <span style="color: #B5CEA8;">0s</span> <span style="color: #B5CEA8;">97ms/step</span>
<span style="color: #CE9178;">Predicted class: 3</span></code></pre>
<p style="font-size: 0.9em; color: #666; margin-top: 10px;">
  This shows the predicted digit for an input image.
</p>

---

<h2 id="code-breakdown" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #8BC34A; padding-bottom: 10px;">
  🧠 Code Breakdown
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  Key parts of the notebook's code structure:
</p>

<h3 style="color: #4CAF50; font-size: 1.8em; margin-top: 25px;">Data Loading & Preprocessing:</h3>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #569CD6;">from</span> <span style="color: #9CDCFE;">tensorflow.keras.datasets</span> <span style="color: #C586C0;">import</span> <span style="color: #9CDCFE;">mnist</span>
<span style="color: #569CD6;">import</span> <span style="color: #9CDCFE;">numpy</span> <span style="color: #C586C0;">as</span> <span style="color: #9CDCFE;">np</span>

<span style="color: #6A9955;"># Loading data</span>
(<span style="color: #9CDCFE;">X_train</span>,<span style="color: #9CDCFE;">y_train</span>),(<span style="color: #9CDCFE;">X_test</span>,<span style="color: #9CDCFE;">y_test</span>) <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">mnist.load_data</span>()

<span style="color: #6A9955;"># Reshaping data for CNN input (add channel dimension)</span>
<span style="color: #9CDCFE;">X_train</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">X_train.reshape</span>((<span style="color: #9CDCFE;">X_train.shape</span>[<span style="color: #B5CEA8;">0</span>],<span style="color: #9CDCFE;">X_train.shape</span>[<span style="color: #B5CEA8;">1</span>],<span style="color: #9CDCFE;">X_train.shape</span>[<span style="color: #B5CEA8;">2</span>],<span style="color: #B5CEA8;">1</span>))
<span style="color: #9CDCFE;">X_test</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">X_test.reshape</span>((<span style="color: #9CDCFE;">X_test.shape</span>[<span style="color: #B5CEA8;">0</span>],<span style="color: #9CDCFE;">X_test.shape</span>[<span style="color: #B5CEA8;">1</span>],<span style="color: #9CDCFE;">X_test.shape</span>[<span style="color: #B5CEA8;">2</span>],<span style="color: #B5CEA8;">1</span>))

<span style="color: #6A9955;"># Normalizing the pixel values to [0, 1]</span>
<span style="color: #9CDCFE;">X_train</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">X_train</span><span style="color: #CE9178;">/</span><span style="color: #B5CEA8;">255.0</span>
<span style="color: #9CDCFE;">X_test</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">X_test</span><span style="color: #CE9178;">/</span><span style="color: #B5CEA8;">255.0</span></code></pre>

<h3 style="color: #4CAF50; font-size: 1.8em; margin-top: 25px;">Building the CNN Model:</h3>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #569CD6;">from</span> <span style="color: #9CDCFE;">tensorflow.keras.models</span> <span style="color: #C586C0;">import</span> <span style="color: #9CDCFE;">Sequential</span>
<span style="color: #569CD6;">from</span> <span style="color: #9CDCFE;">tensorflow.keras.layers</span> <span style="color: #C586C0;">import</span> <span style="color: #9CDCFE;">Dense</span>, <span style="color: #9CDCFE;">Flatten</span>, <span style="color: #9CDCFE;">Conv2D</span>, <span style="color: #9CDCFE;">MaxPool2D</span>

<span style="color: #9CDCFE;">model</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">Sequential</span>()

<span style="color: #6A9955;"># Adding convolution layer</span>
<span style="color: #9CDCFE;">model.add</span>(<span style="color: #9CDCFE;">Conv2D</span>(<span style="color: #B5CEA8;">32</span>,(<span style="color: #B5CEA8;">3</span>,<span style="color: #B5CEA8;">3</span>),<span style="color: #9CDCFE;">activation</span><span style="color: #CE9178;">=</span><span style="color: #CE9178;">'relu'</span>,<span style="color: #9CDCFE;">input_shape</span><span style="color: #CE9178;">=</span>(<span style="color: #B5CEA8;">28</span>,<span style="color: #B5CEA8;">28</span>,<span style="color: #B5CEA8;">1</span>)))

<span style="color: #6A9955;"># Adding pooling layer</span>
<span style="color: #9CDCFE;">model.add</span>(<span style="color: #9CDCFE;">MaxPool2D</span>(<span style="color: #B5CEA8;">2</span>,<span style="color: #B5CEA8;">2</span>))

<span style="color: #6A9955;"># Adding fully connected layers (after flattening)</span>
<span style="color: #9CDCFE;">model.add</span>(<span style="color: #9CDCFE;">Flatten</span>())
<span style="color: #9CDCFE;">model.add</span>(<span style="color: #9CDCFE;">Dense</span>(<span style="color: #B5CEA8;">100</span>,<span style="color: #9CDCFE;">activation</span><span style="color: #CE9178;">=</span><span style="color: #CE9178;">'relu'</span>))

<span style="color: #6A9955;"># Adding output layer</span>
<span style="color: #9CDCFE;">model.add</span>(<span style="color: #9CDCFE;">Dense</span>(<span style="color: #B5CEA8;">10</span>,<span style="color: #9CDCFE;">activation</span><span style="color: #CE9178;">=</span><span style="color: #CE9178;">'softmax'</span>))

<span style="color: #6A9955;"># Compiling the model</span>
<span style="color: #9CDCFE;">model.compile</span>(<span style="color: #9CDCFE;">loss</span><span style="color: #CE9178;">=</span><span style="color: #CE9178;">'sparse_categorical_crossentropy'</span>,<span style="color: #9CDCFE;">optimizer</span><span style="color: #CE9178;">=</span><span style="color: #CE9178;">'adam'</span>,<span style="color: #9CDCFE;">metrics</span><span style="color: #CE9178;">=</span>[<span style="color: #CE9178;">'accuracy'</span>])</code></pre>

<h3 style="color: #4CAF50; font-size: 1.8em; margin-top: 25px;">Training and Prediction:</h3>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #6A9955;"># Fitting the model</span>
<span style="color: #9CDCFE;">model.fit</span>(<span style="color: #9CDCFE;">X_train</span>,<span style="color: #9CDCFE;">y_train</span>,<span style="color: #9CDCFE;">epochs</span><span style="color: #CE9178;">=</span><span style="color: #B5CEA8;">10</span>) <span style="color: #6A9955;"># Or 20, as shown in the notebook for increased epochs</span>

<span style="color: #569CD6;">from</span> <span style="color: #9CDCFE;">tensorflow.keras.utils</span> <span style="color: #C586C0;">import</span> <span style="color: #9CDCFE;">load_img</span>,<span style="color: #9CDCFE;">img_to_array</span>

<span style="color: #6A9955;"># Load and preprocess an external image for prediction</span>
<span style="color: #9CDCFE;">image_path</span><span style="color: #CE9178;">=</span><span style="color: #CE9178;">'/content/download.png'</span> <span style="color: #6A9955;"># Update with your image path</span>
<span style="color: #9CDCFE;">image</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">load_img</span>(<span style="color: #9CDCFE;">image_path</span>,<span style="color: #9CDCFE;">target_size</span><span style="color: #CE9178;">=</span>(<span style="color: #B5CEA8;">28</span>,<span style="color: #B5CEA8;">28</span>),<span style="color: #9CDCFE;">color_mode</span><span style="color: #CE9178;">=</span><span style="color: #CE9178;">'grayscale'</span>)
<span style="color: #9CDCFE;">image_array</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">img_to_array</span>(<span style="color: #9CDCFE;">image</span>)
<span style="color: #9CDCFE;">image_array</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">image_array</span><span style="color: #CE9178;">/</span><span style="color: #B5CEA8;">255.0</span>
<span style="color: #9CDCFE;">image_array</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">image_array.reshape</span>((<span style="color: #B5CEA8;">1</span>,<span style="color: #B5CEA8;">28</span>,<span style="color: #B5CEA8;">28</span>,<span style="color: #B5CEA8;">1</span>))

<span style="color: #6A9955;"># Make prediction</span>
<span style="color: #9CDCFE;">prediction</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">model.predict</span>(<span style="color: #9CDCFE;">image_array</span>)
<span style="color: #9CDCFE;">predicted_class</span><span style="color: #CE9178;">=</span><span style="color: #569CD6;">np.argmax</span>(<span style="color: #9CDCFE;">prediction</span>)
<span style="color: #569CD6;">print</span>(<span style="color: #CE9178;">'Predicted class:'</span>,<span style="color: #9CDCFE;">predicted_class</span>)</code></pre>

---

<h2 id="customization-ideas" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #8BC34A; padding-bottom: 10px;">
  🌈 Customization Ideas
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  Here are some ways to extend and experiment with this project:
</p>
<ul style="list-style-type: none; padding: 0; font-size: 1.1em; color: #444;">
  <li style="margin-bottom: 15px; background-color: #DCEDC8; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #689F38;">🎨 Different CNN Architectures:</strong> Try adding more `Conv2D` and `MaxPool2D` layers, or experiment with different filter sizes and numbers.
  </li>
  <li style="margin-bottom: 15px; background-color: #DCEDC8; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #689F38;">🧪 Different Optimizers/Activations:</strong> Experiment with other optimizers (e.g., SGD, RMSprop) or activation functions (e.g., Leaky ReLU, ELU) in the hidden layers.
  </li>
  <li style="margin-bottom: 15px; background-color: #DCEDC8; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #689F38;">📊 Performance Metrics:</strong> Calculate and display additional evaluation metrics like precision, recall, F1-score, or a classification report.
  </li>
  <li style="margin-bottom: 15px; background-color: #DCEDC8; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #689F38;">🖼️ Data Augmentation:</strong> Implement simple data augmentation techniques (e.g., rotation, shifting, zooming) using `ImageDataGenerator` to improve model generalization.
  </li>
</ul>

---

<h2 id="contribute" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #8BC34A; padding-bottom: 10px;">
  🤝 Contribute
</h2>
<p align="center" style="font-size: 1.2em; color: #555; max-width: 800px; margin: 0 auto; line-height: 1.6;">
  Contributions are always welcome! If you have ideas for improvements, new features, or just want to fix a bug, please feel free to open an issue or submit a pull request. Let’s make handwritten digit recognition even better! 🌟
</p>
<p align="center" style="font-size: 1.2em; color: #555; max-width: 800px; margin: 15px auto 0; line-height: 1.6;">
  Star this repo if you find it helpful! ⭐
</p>
<p align="center" style="font-size: 1em; color: #777; margin-top: 30px;">
  Created with 💖 by the Chirag
</p>
