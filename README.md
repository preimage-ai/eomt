# Your ViT is Secretly an Image Segmentation Model (CVPR 2025)

*[Tommie Kerssies](https://tommiekerssies.com)¹, [Niccolò Cavagnero](https://scholar.google.com/citations?user=Pr4XHRAAAAAJ)²<sup>,</sup>\*, [Alexander Hermans](https://scholar.google.de/citations?user=V0iMeYsAAAAJ)³, [Narges Norouzi](https://scholar.google.com/citations?user=q7sm490AAAAJ)¹, [Giuseppe Averta](https://www.giuseppeaverta.me/)², [Bastian Leibe](https://scholar.google.com/citations?user=ZcULDB0AAAAJ)³, [Gijs Dubbelman](https://scholar.google.nl/citations?user=wy57br8AAAAJ)¹, [Daan de Geus](https://ddegeus.github.io)¹<sup>,</sup>³*

<sup>1</sup> Eindhoven University of Technology  
<sup>2</sup> Polytechnic of Turin  
<sup>3</sup> RWTH Aachen University  
<sup>\*</sup> _Work done while visiting RWTH Aachen University._


---

📄 **Paper**: [arXiv](https://arxiv.org/abs/2503.19108)  
💻 **Code**: Coming soon  
👁️ **Project page**: [https://tue-mps.github.io/eomt](https://tue-mps.github.io/eomt)  
🛎️ **Stay updated**: [Watch the repository](https://github.com/tue-mps/eomt/subscription)  
🐞 **Questions or issues?** [Open a GitHub issue](https://github.com/tue-mps/eomt/issues)  
📬 **Contact**: t.kerssies[at]tue[dot]nl

---
Welcome to the official code repository of "**Your ViT is Secretely an Image Segmentation Model**".

In this paper, we show that task-specific components for image segmentation with Vision Transformers (ViTs) become increasingly redundant as model size and pre-training are scaled up. By removing all such components, we introduce the **Encoder-only Mask Transformer (EoMT)**, a segmentation model that purely uses a plain ViT, revealing that _your ViT is secretly an image segmentation model_.

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

**Coming soon**

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
