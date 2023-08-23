# Deepfake Detection Using OpenFace2 and LSTM
This project aims to develop a minimum viable product for detecting deepfake videos using the LSTM architecture and OpenFace2. The project is a work in progress.

<h2>Project Overview</h2>
<p>The project is built to detect deepfake videos using LSTM in combination with various feature extraction techniques. Data sources are explained below. The initial data source was Celeb-DF. Later data from Kaggle was incorporated. The scope of the project is to detect deepfake videos using the intensity of action units and how they represent the difference between real and fake videos.</p>

<h2>Dependencies</h2>
<ul>
  <li>Python 3</li>
  <li>PyTorch</li>
  <li>OpenFace 2.0.0</li>
  <li>OpenCv4</li>
</ul>

<h2>OpenFace 2</h2>
<p>Open Face 2 was installed on a windows computer using the Windows installation guide on the OpenFace 2 <a href="https://github.com/TadasBaltrusaitis/OpenFace/wiki">WIKI</a>. The OpenFace 2 Windows implementation provides a GUI. This GUI was used to initially load 1706 videos of both fake and real and extract features from these videos.</p>
<b><p>Current TODO: Expand the dataset using other sources</p></b>

<h2>Data Preprocessing</h2>
