# Your ViT is Secretly an Image Segmentation Model (CVPR 2025)

*Tommie Kerssies¹, Niccolò Cavagnero²\*, Alexander Hermans³, Narges Norouzi¹, Giuseppe Averta², Bastian Leibe³, Gijs Dubbelman¹, Daan de Geus¹³*

<sup>1</sup> Eindhoven University of Technology  
<sup>2</sup> Polytechnic of Turin  
<sup>3</sup> RWTH Aachen University  
<sup>\*</sup> _Work done while visiting RWTH Aachen University._

---

📄 **Paper**: Coming soon  
💻 **Code**: Coming soon  
👁️ **Project page**: [https://tue-mps.github.io/eomt](https://tue-mps.github.io/eomt)  
🛎️ **Stay updated**: [Watch the repository](https://github.com/tue-mps/eomt/subscription)  
🐞 **Questions or issues?** [Open a GitHub issue](https://github.com/tue-mps/eomt/issues)  
📬 **Contact**: t.kerssies[at]tue[dot]nl

---

## Installation

```bash
conda create -n EoMT python==3.12
conda activate EoMT
pip install -r requirements.txt
```

---

## Data preparation

Download the following datasets to train and test EoMT models. The code will access the datasets at the location specified by the **root** parameter.

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
```bash
wget https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget https://www.cityscapes-dataset.com/file-handling/?packageID=3
```

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
