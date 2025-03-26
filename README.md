# Your ViT is Secretly an Image Segmentation Model (CVPR 2025)

*[Tommie Kerssies](https://tommiekerssies.com)¹, [Niccolò Cavagnero](https://scholar.google.com/citations?user=Pr4XHRAAAAAJ)²<sup>,</sup>\*, [Alexander Hermans](https://scholar.google.de/citations?user=V0iMeYsAAAAJ)³, [Narges Norouzi](https://scholar.google.com/citations?user=q7sm490AAAAJ)¹, [Giuseppe Averta](https://www.giuseppeaverta.me/)², [Bastian Leibe](https://scholar.google.com/citations?user=ZcULDB0AAAAJ)³, [Gijs Dubbelman](https://scholar.google.nl/citations?user=wy57br8AAAAJ)¹, [Daan de Geus](https://ddegeus.github.io)¹<sup>,</sup>³*

<sup>1</sup> Eindhoven University of Technology  
<sup>2</sup> Polytechnic of Turin  
<sup>3</sup> RWTH Aachen University  
<sup>\*</sup> _Work done while visiting RWTH Aachen University._


---
Welcome to the official code repository of "**Your ViT is Secretly an Image Segmentation Model**".

📄 **Paper**: [arXiv](https://arxiv.org/abs/2503.19108)  
💻 **Code**: Coming soon  
👁️ **Project page**: [https://tue-mps.github.io/eomt](https://tue-mps.github.io/eomt)  
🛎️ **Stay updated**: [Watch the repository](https://github.com/tue-mps/eomt/subscription)  
🐞 **Questions or issues?** [Open a GitHub issue](https://github.com/tue-mps/eomt/issues)  
📬 **Contact**: t.kerssies[at]tue[dot]nl

---
## Abstract

Vision Transformers (ViTs) have shown remarkable performance and scalability across various computer vision tasks. To apply single-scale ViTs to image segmentation, existing methods adopt a convolutional adapter to generate multi-scale features, a pixel decoder to fuse these features, and a Transformer decoder to make predictions.

In this paper, we show that the inductive biases introduced by these task-specific components can instead be learned by the ViT itself, given sufficiently large models and extensive pre-training. Based on these findings, we introduce the Encoder-only Mask Transformer (EoMT), which repurposes the plain ViT architecture for efficient image segmentation.

With large-scale models and pre-training, EoMT achieves segmentation accuracy similar to state-of-the-art methods while being significantly faster due to its architectural simplicity.

---

## Installation

```bash
conda create -n EoMT python==3.12
conda activate EoMT
pip install -r requirements.txt
```

---

## Data preparation

Download the following datasets for training and testing EoMT models. Downloading is optional and depends on which datasets you plan to use. There is no need to unzip the folders. The code will access the datasets directly from the location specified by the **--root** parameter.

**COCO**
```bash
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
```

**ADE20K**
```bash
wget http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip
wget http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
tar -xf annotations_instance.tar
zip -r -0 annotations_instance.zip annotations_instance/
rm -rf annotations_instance.tar
rm -rf annotations_instance
```

**Cityscapes**

Please refer to this [GitHub repository](https://github.com/cemsaz/city-scapes-script) and download the .zip files for packageID=1 and packageID=3.

---

## Usage

**Coming soon**

---

## Model Zoo

### Panoptic Segmentation

#### COCO

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Resolution</th>
<th valign="bottom">FPS</th>
<th valign="bottom">PQ</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 640x640 -->
<tr><td align="left"><a href="#">EoMT-L</a></td>
<td align="center">640x640</td>
<td align="center">128</td>
<td align="center">56.0</td>
<td align="center"><a href="#">Coming soon</a></td>
</tr>
<!-- ROW: EoMT-L 1280x1280 -->
<tr><td align="left"><a href="#">EoMT-L</a></td>
<td align="center">1280x1280</td>
<td align="center">30</td>
<td align="center">58.3</td>
<td align="center"><a href="#">Coming soon</a></td>
</tr>
<!-- ROW: EoMT-g 640x640 -->
<tr><td align="left"><a href="#">EoMT-g</a></td>
<td align="center">640x640</td>
<td align="center">55</td>
<td align="center">57.0</td>
<td align="center"><a href="#">Coming soon</a></td>
</tr>
<!-- ROW: EoMT-g 1280x1280 -->
<tr><td align="left"><a href="#">EoMT-g</a></td>
<td align="center">1280x1280</td>
<td align="center">12</td>
<td align="center">59.2</td>
<td align="center"><a href="#">Coming soon</a></td>
</tr>
</tbody></table>

#### ADE20K

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Resolution</th>
<th valign="bottom">FPS</th>
<th valign="bottom">PQ</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 640x640 -->
<tr><td align="left"><a href="#">EoMT-L</a></td>
<td align="center">640x640</td>
<td align="center">128</td>
<td align="center">50.6</td>
<td align="center"><a href="#">Coming soon</a></td>
</tr>
<!-- ROW: EoMT-L 1280x1280 -->
<tr><td align="left"><a href="#">EoMT-L</a></td>
<td align="center">1280x1280</td>
<td align="center">30</td>
<td align="center">51.7</td>
<td align="center"><a href="#">Coming soon</a></td>
</tr>
<!-- ROW: EoMT-g 640x640 -->
<tr><td align="left"><a href="#">EoMT-g</a></td>
<td align="center">640x640</td>
<td align="center">55</td>
<td align="center">51.3</td>
<td align="center"><a href="#">Coming soon</a></td>
</tr>
<!-- ROW: EoMT-g 1280x1280 -->
<tr><td align="left"><a href="#">EoMT-g</a></td>
<td align="center">1280x1280</td>
<td align="center">12</td>
<td align="center">52.8</td>
<td align="center"><a href="#">Coming soon</a></td>
</tr>
</tbody></table>

### Semantic Segmentation

#### Cityscapes

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Resolution</th>
<th valign="bottom">FPS</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 1024x1024 -->
<tr><td align="left"><a href="#">EoMT-L</a></td>
<td align="center">1024x1024</td>
<td align="center">25</td>
<td align="center">84.2</td>
<td align="center"><a href="#">Coming soon</a></td>
</tr>
</tbody></table>

#### ADE20K

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Resolution</th>
<th valign="bottom">FPS</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 512x512 -->
<tr><td align="left"><a href="#">EoMT-L</a></td>
<td align="center">512x512</td>
<td align="center">92</td>
<td align="center">58.4</td>
<td align="center"><a href="#">Coming soon</a></td>
</tr>
</tbody></table>

### Instance Segmentation

#### COCO

#### Cityscapes

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Resolution</th>
<th valign="bottom">FPS</th>
<th valign="bottom">AP</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 640x640 -->
<tr><td align="left"><a href="#">EoMT-L</a></td>
<td align="center">640x640</td>
<td align="center">128</td>
<td align="center">45.2</td>
<td align="center"><a href="#">Coming soon</a></td>
</tr>
<!-- ROW: EoMT-L 1280x1280 -->
<tr><td align="left"><a href="#">EoMT-L</a></td>
<td align="center">1280x1280</td>
<td align="center">30</td>
<td align="center">48.8</td>
<td align="center"><a href="#">Coming soon</a></td>
</tr>
</tbody></table>


---

## Citation
If you find this project helpful for your research, please consider citing the following BibTeX entry.

```BibTeX
@inproceedings{kerssies2025eomt,
  author    = {Kerssies, Tommie and Cavagnero, Niccolò and Hermans, Alexander and Norouzi, Narges and Averta, Giuseppe and Leibe, Bastian and Dubbelman, Gijs and de Geus, Daan},
  title     = {Your ViT is Secretly an Image Segmentation Model},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
}
```
