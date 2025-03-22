# Gorgeous: Creating Narrative-Driven Makeup Ideas via Image Prompt 

---

![](docs/assets/teaser.png)

Environment Setup:
1. conda create -n gorgeous python=3.10
2. conda activate gorgeous
3. pip install -r requirements.txt

Context Learning:
1. bash train_textualinversion.sh `<dataset_name>` `<initializer_token>`
2. `<dataset_name>` see "makeup_assets/`<dataset_name>`"
3. Note that the dataset in `makeup_assets` is for research purpose only.

Gradio:
1. python app.py
