## DeepFaceLab and DeepFaceLive on Stable Diffusion

This is an implementation of iperov's DeepFaceLab and DeepFaceLive in Stable Diffusion Web UI by AUTOMATIC1111.

Note: DeepFaceLive is currently fully implemented while DeepFaceLab is in the works.

This can be installed as a plugin for https://github.com/AUTOMATIC1111/stable-diffusion-webui

DeepFaceLive is implemented as a script which can be accessed on txt2img and img2img tabs.

DeepFaceLab has a separate tab and controls to manage workspaces and train custom models.

## Requirements

- Python 3.10

## Additional implementations

In order to implement DeepFaceLab which at the moment of writing uses Python 3.6 and DeepFaceLive which uses Python 3.7 the following modifications had to be made in order to modernize it to run on Python 3.10

- Implemented xlib in DeepFaceLive had been updated for Collections import to work

## Use cases

- You can easily face swap any face in stable diffusion with the one that you want, with a combination of DeepFaceLab to create your model and DeepFaceLive to implement the model to be used in stable diffusion generating process.
- Enhance and make more stable and person specific the output of faces in stable diffusion.
- It's up to you

## TODO
- Implement DeepFaceLab.

## References

- Stable Diffusion Web UI by AUTOMATIC1111: https://github.com/AUTOMATIC1111/stable-diffusion-webui
- DeepFaceLab by iperov: https://github.com/iperov/DeepFaceLab
- DeepFaceLive by iperov: https://github.com/iperov/DeepFaceLive
- Detection Detailer by dustysys: https://github.com/dustysys/ddetailer
