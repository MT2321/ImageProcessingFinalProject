# Image Processing FinalProject
The goal of this project is to identify and classify a certain type of tv ad.

<center>
<img src="/assets/imgs/25.jpg" width="300)" />
</center>
In particular we desire to extract and classify those rectangular ads that can be seen at the bottom.

<br></br>
Here we present an overview of the project
```mermaid
graph LR

subgraph Box Detection
    A[Input Image] --> B(Bounding Box Extraction)
    B-->C[Image Crop]
end

C-->F
C-->G
subgraph Clustering
    D[NNClassifier]
end

subgraph Feature Extraction
F[Color Histogram] --> X
G[VGG16 Latent Space] -->X
X[Concat features]-->D
end
```


Let's begin by understanding how does the **bounding box extraction** process works.
```mermaid

graph LR
A[input image]-->B(Blur/LPF) --> C(Canny Edge extraction)

-->D(Dilate - Erode) --> E(Vertical/Horizontal lines) --> F(Box Detection)-->G[Bounding Box]

```

In the first step we apply a **blur** kernel to filter out the high frequency noise present in low quality images. This helps in the following step to avoid noise amplification.
<center>
<img src="/results/step 1 blur.jpg" width="300)" />
</center>
This step is followed by performing classic edge detection with Canny

<center>
<img src="/results/step 2 canny.jpg" width="300)" />
</center>
It can already be seen that this image contains a great ammount of edges. Most of them are not relevant to our case. It is worth noting that the straight edges are far from perfect.

<br></br>

We continue by performing a **dilate-erode** operation, also known as **close**. This is applied in order to fill in the gaps that some contours may have. As the images we are dealing with of very low quality this steps provides stronger, more continous edges to apply further processing.

<center>
<img src="/results/step 3 dilate.jpg" width="300)" />
</center>
<center>
<img src="/results/step 4 erode.jpg" width="300)" />
</center>