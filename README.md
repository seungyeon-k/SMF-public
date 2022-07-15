# A Statistical Manifold Framework for Point Cloud Data
The official repository for \<A Statistical Manifold Framework for Point Cloud Data\> (Yonghyeon Lee*, Seungyeon Kim*, Jinwon Choi, and Frank C. Park, ICML 2022).

<sup>\*</sup> The two lead co-authors contributed equally.

> This paper proposes a new Riemannian geometric structure for the space of point cloud data, using the theory of statistical manifold and information geometry, with applications to point cloud autoencoders.

- *[Paper](https://proceedings.mlr.press/v162/lee22d/lee22d.pdf)* 
- 5-mins video (TBA)
- *[Slides](https://docs.google.com/presentation/d/1NvPuEoXqi93NKw13b4xzb84gT_viRG8O/edit?usp=sharing&ouid=102571685065826150745&rtpof=true&sd=true)*
- *[Poster](https://drive.google.com/file/d/1NuyQaG-g3zwWQEl6qS5tl5_H3oKsEzcK/view?usp=sharing)*  

## Preview
### Statistical Manifold and Information Geometry
<!-- <center>
<div class="imgCollage">
<span style="width: 50.0%"><img src="./figures/statistical_manifold.PNG" width="250" height="100"/></span>
<span style="width: 50.0%"><img src="./figures/vector_field.PNG" width="250" height="190"/> </span>
</div>
  <I>Figure 1: De-noising property of the NRAE (Left: Vanilla AE, Middle: NRAE-L, Right: NRAE-Q). </I>
</center> -->
<center>
<img src="./figures/vector_field.PNG" alt="drawing" width="400"/>
</center>
<I> Figure 1: Two moving point clouds with different velocity
matrices. </I>

### Interpolation
![interpolation](figures/interpolation.PNG)
<I> Figure 2: Left: Latent space with linear and geodesic interpolants. The orange interpolants connect a wide cylinder to a tall cylinder, while the magenta interpolants connect a cylinder to a cone. Linear interpolants and geodesic interpolants under the Euclidean and info-Riemannian metrics are drawn as dotted, dashed, and solid lines, respectively. Right: Generated point clouds from those interpolants. </I>

### Regularization
![regularization](figures/regularization.PNG)
<I>Figure 3: From left to right: latent spaces with equidistant
ellipse $\{z|(z-z^*)^T G(z^*) (z-z^*) = 1\}$ centered on
some selected points $z^*$, Gaussian Mixture Model
(GMM) fitting results, generated samples from the GMM, and the heat map of the pairwise Euclidean distances in the latent space of all test data. The upper figure is a vanilla autoencoder trained without regularization, while the lower figure is trained with regularization (using the proposed info-Riemannian metric). </I>

## Progress

- [ ] README update
- [ ] Training script (`train.py`)
- [ ] Dataset upload
- [ ] Pre-trained model upload
- [ ] Data generation script (`data_generation.py`)
- [ ] Point cloud interpolation script (`interpolation.py`)
- [ ] Regularization effect visualization script (`regularization.py`)

## Requirements
### Environment
The project is developed under a standard PyTorch environment.
- python 3.8.8
- torch 1.8.0

### Datasets
Datasets should be stored in `datasets/` directory. Datasets can be set up in one of two ways.
- Run the data generation script:
preparing...
<!-- ```
python data_generator.py
``` -->

- Download through the [Google drive link](https://drive.google.com/drive/folders/1NuGq2LtWG627r9BNPzb1EegUuIvPUzDr?usp=sharing)

### (Optional) Pretrained model
Pre-trained models should be stored in `pretrained/`. The pre-trained models are provided through the [Google drive link](https://drive.google.com/drive/folders/1NuYIfyU6kVQ09qPR6rONWrernKMps_FX?usp=sharing).
When set up, the pretrained directory should look like as follows.

## Running 
### Training
preparing...

### Interpolation
preparing...

### Regularization
preparing...

### Tips for playing with the code
Preparing...

## Citation
If you found this library useful in your research, please consider citing:
```
@inproceedings{lee2022statistical,
  title={A Statistical Manifold Framework for Point Cloud Data},
  author={Lee, Yonghyeon and Kim, Seungyeon and Choi, Jinwon and Park, Frank},
  booktitle={International Conference on Machine Learning},
  pages={12378--12402},
  year={2022},
  organization={PMLR}
}
```