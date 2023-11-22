# Deepfake Detection Using OpenFace2 and LSTM
This project aims to develop a minimum viable product for detecting deepfake videos using the LSTM architecture and OpenFace2. The project is a work in progress.

<h2>Project Overview</h2>
<p>The project is built to detect deepfake videos using LSTM in combination with OpenFace2 for facial feature extraction. Celeb-DF is the primary data source used in training the model. Features 
extraction from the videos are used to train the model in the form of csv files. The project currently achieves a validation accuracy of 77% and we are hoping to improve the model and incorporate more techniques into it.</p>

<h2>Dependencies</h2>
<ul>
  <li>Python 3</li>
  <li>PyTorch</li>
  <li>OpenFace 2.0.0</li>
  <li>OpenCv4</li>
</ul>

<h2>OpenFace 2</h2>
<p><a href="https://github.com/TadasBaltrusaitis/OpenFace">OpenFace 2</a> is used in the project for facial feature extraction. It is the main feature source of the project. OpenFace 2 can be installed by cloning their <a href="https://github.com/TadasBaltrusaitis/OpenFace">repository</a> and following the installation guide in their <a href="https://github.com/TadasBaltrusaitis/OpenFace/wiki#installation">wiki</a>. The Ubuntu installation was followed for this project. The "FeatureExtraction" executable file is used to extract features in a Python script before being preprocessed and used for training.</p>
<p>Use the installation steps for the respective operating system in which you are going to train the model and ensure that the openface directory is in the root of the project.</p>

<h2>Preparing new Videos</h2>
<p>The script folder contains the scripts required to preprocess, prepare and label new video data. The code does not currently load the latest checkpoint but you can easily add this before starting the training loop.</p>
<p>First step is to extract features from your videos using the feature_extraction.py file.</p>
<ul>
  <li>--openface_exe : the path to the openface feature extraction binary file. This would normally be in ./build/bin of the open face directory.</li>
  <li>--input_dir : the directory where your video files reside.</li>
  <li>--output_dir : the directory to which you wish to save the extracted features.</li>
</ul>
<p>Simply run the script with the above parameters and the extracted features will be saved to the output directory.</p>
<p>Once you have extracted the features, you can run the prepocessor.py script. This script has several functions which were used in early stages of the project.</p>
