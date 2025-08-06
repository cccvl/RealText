# RealText: Realistic Text Image Generation based on Glyph and Scene Aware Inpainting (ACM MM2025)
## Installation
```bash
git clone https://github.com/cccvl/RealText
cd ./RealText
conda create -n RealText python=3.10.0
conda activate RealText
chmod +x bash.sh
./bash.sh
```
## Checkpoints
Download the checkpoint file and put it in ./checkpoints.
The pre-trained models used by RealText are all open-source T2I models or Controlnet models. The specific checkpoint file sources can be found in Table 1 of our paper.
## Fonts
Put the .ttf file in ./fonts.
## Gui
```bash
python gui.py --gb_model "Flux.1" --gw_model "SDXL-x"
```
Please check the the model loading in gui.py and confirm that the corresponding checkpoint file has been downloaded. If you need to add a new model, the code needs to be modified.
