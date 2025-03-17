[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13334951.svg)](https://doi.org/10.5281/zenodo.13334951)
![Static Badge](https://img.shields.io/badge/BI_ADD-a?style=flat&label=arXiv&color=%23B31B1B&link=https%3A%2F%2Farxiv.org%2Fabs%2F2503.11529)
## BI-ADD (Bottom-up Iterative Anomalous Diffusion Detector)

> [!IMPORTANT]  
> Requirements </br>
> - TensorFlow 2.14
> - Python 3.10
> - latest version of [scikit-learn](https://scikit-learn.org/stable/)[^3]
> - latest version of [scikit-image](https://scikit-image.org/docs/stable/user_guide/install.html)[^4]
> - Pre-trained [models](https://github.com/JunwooParkSaribu/BI_ADD/tree/main/models)


<b>BI-ADD</b> detects changepoints at single molecular trajectory level which follows fBm with two properties, Anomalous exponent(alpha) and Generalized diffusion coefficient(K), on different scenarios.</br>
For the details of data and scenarios of trjaectories, please check Andi2 Challenge[^1][^2].</br>
The trajectory prediction from video is performed with <b>[FreeTrace](https://github.com/JunwooParkSaribu/FreeTrace)</b></br>


<table border="0"> 
        <tr> 
            <td>On simulated trajectories</td> 
        </tr>
        <tr> 
            <td><img src="https://github.com/JunwooParkSaribu/BI_ADD/blob/main/tmps/imgs/alpha_test0.gif" width="780" height="390"></td> 
        </tr>
</table>

<img src="https://github.com/JunwooParkSaribu/AnDi2_SU_FIONA/blob/main/tmps/imgs/foot.png" width="232" height="64"></br>


<h3> To run the program on your device </h3>

1. Clone the repository on your local device.</br>
2. Download pre-trained [*models*](https://drive.google.com/file/d/1WF0eW8Co23-mKQiHNH-KHHK_lJiIW-WC/view?usp=sharing), place the *models* folder inside of *BI_ADD* folder.</br>
3. Place trajectory csv files(traj_idx, frame, x, y) inside *inputs* folder.</br>
4. Run *run.py* via python.</br>

<h3> Contacts </h3>
junwoo.park@sorbonne-universite.fr</br>

<h3> References </h3>

[^1]: [AnDi datasets](https://doi.org/10.5281/zenodo.10259556)
[^2]: Quantitative evaluation of methods to analyze motion changes in single-particle experiments, Gorka Muñoz-Gil et al. 2024 [https://arxiv.org/abs/2311.18100](https://arxiv.org/abs/2311.18100)
[^3]: Convolutional LSTM Network, A Machine Learning Approach for Precipitation Nowcasting, Xingjian Shi et al. 2015 [https://arxiv.org/abs/1506.04214](https://arxiv.org/abs/1506.04214)
[^4]: Chenouard, N., Smal, I., de Chaumont, F. et al. [Objective comparison of particle tracking methods](https://doi.org/10.1038/nmeth.2808). Nat Methods 11, 281–289 (2014).