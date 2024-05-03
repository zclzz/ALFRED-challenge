# Using Natural Language to Command Embodied Artificial Intelligence

[Strategic Task Execution: Integrating Large Language Models, POMDP, and Customized Reordering Policy for Sequence Selection
](https://zclzz.github.io/alfred_blog/)<br />
[Chao Zhou Lim](https://zclzz.github.io/portfolio/), advised by Dr. Kim Jung-Jae and Dr. Goh Shen Tat<br />

![example](./miscellaneous/coffee_edited-ezgif.com-optimize.gif)



## Setting up the environment

Please follow [Prompter](https://github.com/hitachi-rd-cv/prompter-alfred) repository and directions for setting up the environment which my system is based on. My code is a fork of Prompter 

PLease make sure that you are using **Python 3.6** and **AI2THOR ver. 2.1.0**.

### Dataset


Please download the [dataset](https://github.com/askforalfred/alfred/tree/master/data) from here.

   - Follow the instructions in the [FILM](https://github.com/soyeonm/FILM) repo to get started ensuring that your files are structured in a similar manner

   - The last step asks to create a soft link from `json_2.1.0` to `alfred_data_all`. However, in my experience I had to link from `alfred_feat_2.1.0`.
     ```bash
     # before
     ln -s $ALFRED_ROOT/data/json_2.1.0 $FILM/alfred_data_all
     
     # after
     ln -s $ALFRED_ROOT/data/json_feat_2.1.0/ $FILM/alfred_data_all/json_2.1.0
     ```
     
   - After this step, `alfred_data_all` directory should look like this:

      ```bash
   alfred_data_all
      └── json_2.1.0
       ├── tests_unseen
          ├── tests_seen
          ├── valid_unseen
          ├── tests_seen
          ├── trial_T2019...
          └── ...
      ```


## Training / Evaluation

My agent relies on the pre-trained segmentation models, however, the large language models for language understanding were self-trained and fine-tuned.

If you plan on using some pre-trained language models trained by FILM, please follow the instructions below:

[Download Trained models](https://github.com/soyeonm/FILM#download-trained-models)

   1. Download "Pretrained_Models_FILM" from [this link](https://drive.google.com/file/d/1mkypSblrc0U3k3kGcuPzVOaY1Rt9Lqpa/view?usp=sharing) kindly provided by FILM's author

   2. Relocate the downloaded models to the correct directories

      ```
      mv Pretrained_Models_FILM/maskrcnn_alfworld models/segmentation/maskrcnn_alfworld
      mv Pretrained_Models_FILM/depth_models models/depth/depth_models
      mv Pretrained_Models_FILM/new_best_model.pt models/semantic_policy/best_model_multi.pt
      ```

## Acknowledgements

My agent's system is based on [Prompter's repository](https://github.com/hitachi-rd-cv/prompter-alfred) and [FILM's repository](https://github.com/soyeonm/FILM).



