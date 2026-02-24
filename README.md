# Gorgeous: Creating Narrative-Driven Makeup Ideas via Image Prompt (Official Site)

Updated in May. 05, 2025 (Published at [Multimedia Tools and Applications, Springer](http://cs-chan.com/doc/MTAP2025.pdf))

Released on February 20, 2025

---

![](docs/assets/teaser.png)

## :rocket: Introduction
Gorgeous is a diffusion-based makeup generator that turns any image prompt (like fire or moonlight) into creative, narrative-driven makeup on a face, rather than simply copying an existing style. It first learns what makeup is, then uses pseudo-paired training (via face parsing and content–style disentanglement) to learn from bare-face images, and finally applies a context-guided inpainting method to produce region-specific, real-world-ready makeup designs.
  
From inspiration to personalized beauty in seconds to unlock scalable, AI-powered creativity for brands, artists, and consumers.

## :hammer_and_wrench: Getting Start
Environment Setup:
1. conda create -n gorgeous python=3.10
2. conda activate gorgeous
3. pip install -r requirements.txt

Context Learning:
1. bash train_textualinversion.sh `<dataset_name>` `<initializer_token>`
2. `<dataset_name>` see "makeup_assets/`<dataset_name>`" These are few examples of makeup ideas we trained for research purpose.
3. Note that the dataset in `makeup_assets` is for research purpose only.

Gradio:
1. python app.py

## Citation

```bibtex
@article{sii2025gorgeous,
  title={Gorgeous: Creating narrative-driven makeup ideas via image prompts},
  author={Sii, Jia Wei and Chan, Chee Seng},
  journal={Multimedia Tools and Applications},
  pages={1-22},
  year={2025},
  publisher={Springer}
}
```

## Feedback
Suggestions and opinions on this work are greatly welcomed. Please contact the authors by sending an email to
`adrianasjw at hotmail.com` or `cs.chan at um.edu.my`.

## License and Copyright
The project is open source under BSD-3 license (see the ``` LICENSE ``` file). 

For commercial purpose usage, please contact Dr. Chee Seng Chan at `cs.chan at um.edu.my`

&#169;2024 Universiti Malaya.
