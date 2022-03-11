# Image Processing FinalProject
The goal of this project is to identify and classify a certain type of tv ad.

<p align="center">
<img src="/assets/imgs/25.jpg" width="300" />
</p>
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
# Bounding Box Extraction
Let's begin by understanding how does the **bounding box extraction** process works.
```mermaid

graph LR
A[input image]-->B(Blur/LPF) --> C(Canny Edge extraction)

-->D(Dilate - Erode) --> E(Vertical/Horizontal lines) --> F(Box Detection)-->G[Bounding Box]

```

In the first step we apply a **blur** kernel to filter out the high frequency noise present in low quality images. This helps in the following step to avoid noise amplification.
<p align="center">
<img src="/results/step 1 blur.jpg" width="300" />
</p>
This step is followed by performing classic edge detection with Canny

<p align="center">
<img src="/results/step 2 canny.jpg" width="300" />
</p>
It can already be seen that this image contains a great ammount of edges. Most of them are not relevant to our case. It is worth noting that the straight edges are far from perfect.

<br></br>

We continue by performing a **dilate-erode** operation, also known as **close**. This is applied in order to fill in the gaps that some contours may have. As the images we are dealing with of very low quality this steps provides stronger, more continous edges to apply further processing.

<p align="center">
<img src="/results/step 3 dilate.jpg" width="300" />
</p>
<p align="center">
<img src="/results/step 4 erode.jpg" width="300" />
</p>

## Let's wrap it up

<p align="center">
<img src="/results/step 4 vh lines.jpg" width="300" />
</p>

Removing all the unwanted edges allows us to focus on only the polygons of interest. We can clearly see two rectangles where the ads are.

After applying OpenCVs tool to find polygons and discarding those which are really small. We get

<p align="center">
<img src="/results/final_result.jpg" width="300" />
</p>

Finally, just crop the images. We made sure to keep track of the parent-child relationship between images by storing them into a dataframe.
<p align="center">
<img src="/results/individual_spots/25_0.jpg" width="200" />
<img src="/results/individual_spots/25_1.jpg" width="200" />
</p>


# Feature extraction
In order to clusterize the ads we need to extract some kind of feature vector that allows us to compare them.
Such feature vector, in our case, is composed of the concatention of the **feature space output** of a pretrained VGG11 model and a **color histogram**.
In this manner we are to combine both classic and sota approached towards the computation of a feature vector.

Technically it is not necessary to to include the color histogram but the course requires to solve this problem with a classic approach.
# Clustering and Classification

We trained a Nearest Neighbors classifier in order to find where does the same ad appear.

The euclidean metric was used to determine similarity.



<p align="center">
<img src="/results/clustered/13.jpg" width="500" />
</p>

In this example, we see the **DENIM MARKET** ad appearing in all those images