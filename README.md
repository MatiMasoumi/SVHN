# Street View House Numbers (SVHN) Classification ## Overview This project involves building deep
learning models to classify house numbers from the **Street View House Numbers (SVHN)** dataset. The dataset consists of real-world images of house numbers captured in urban environments.
The classification models used in this project are: - **VGG16** (pretrained on ImageNet) - **ResNet50** (pretrained on ImageNet) ## Dataset - The **SVHN** dataset contains images of digits 
(0–9) extracted from street view imagery. - Each image contains one or more digits, but for this project, we focused on single-digit classification. - **Challenge**: Due to low-quality internet
connections in Iran, accessing high-resolution images from the dataset was problematic. As a result, some data samples had lower resolution or quality, impacting model performance.
## Models ### 1. VGG16 - Pretrained on ImageNet and fine-tuned on the SVHN dataset. - **Evaluation Results**: - **Loss**: `0.1944` - **Accuracy**: `94.56%` ### 2.
ResNet50 - Pretrained on ImageNet and fine-tuned on the SVHN dataset. - **Evaluation Results**: - **Loss**: `0.4454` - **Accuracy**: `87.67%` ### Observations -
The VGG16 model outperformed ResNet50 in terms of both accuracy and loss on the SVHN test set. ## Project Workflow 1. **Data Preparation**: - The dataset was preprocessed by
resizing the images to fit the input dimensions of the models. - Images were normalized to improve model convergence. 2. **Training**: - Both models were trained using a GPU for
faster computation. - Standard data augmentation techniques were applied to improve model robustness. 3. **Evaluation**: - Models were evaluated on a separate test set, and key
metrics such as accuracy and loss were recorded. 4. **Visualization**: - Predictions from the models were visualized to assess performance qualitatively. ## Requirements To 
this project, install the required libraries: ```bash pip install tensorflow numpy matplotlib
