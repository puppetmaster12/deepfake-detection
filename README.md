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

<h2>Data Preprocessing</h2>
