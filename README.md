# Breast Tumor Segmentation in Ultrasound Images Using U-Net Architectures

This repository contains code for training and evaluating four different U-Net-based deep learning models to perform breast tumor segmentation on ultrasound images. The project explores the use of standard and advanced variations of U-Net models, all implemented in PyTorch, to identify tumors in ultrasound scans.

## Models

The following models were trained and evaluated for tumor segmentation:
1. **U-Net** - Standard U-Net model.
2. **Recurrent U-Net** - U-Net with recurrent connections added to improve feature retention.
3. **Attention U-Net** - U-Net with an attention mechanism to enhance focus on relevant regions.
4. **Recurrent Attention U-Net** - A combination of recurrent connections and attention mechanisms.

## Dataset

The dataset used for this project is described in the following paper:

Citation: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. *Data in Brief.* 2020 Feb;28:104863. DOI: [10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863).

This dataset is relatively small, containing only 780 images. Given the limited size, training models from scratch can lead to overfitting and suboptimal performance. To overcome this, we leveraged both custom architectures and pre-trained models.

## Results

The accuracy of each model in segmenting tumors is as follows:

| Model                     | Accuracy |
|---------------------------|----------|
| U-Net                     |   96%    |
| Recurrent U-Net           |   84%    |
| Attention U-Net           |   96%    |
| Recurrent Attention U-Net |   91%    |

Interestingly, the simpler U-Net model achieved the best performance, likely due to the limited dataset size. Neither recurrence nor attention mechanisms appeared to improve segmentation accuracy, which may indicate that more complex architectures require larger datasets for optimal performance.

## Observations

Despite the advanced architectures of the Recurrent U-Net, Attention U-Net, and Recurrent Attention U-Net models, the standard U-Net achieved the highest segmentation accuracy, potentially due to the limited dataset size. Future work may involve retraining these models on a larger dataset to evaluate the true potential of recurrence and attention mechanisms for tumor segmentation.

## License

This project is licensed under the MIT License.

## Acknowledgments

This project was developed as a continuation of the classification project using the same breast ultrasound d
