training.ipynb contains the code used for training the two models.

evaluate.ipynb contains the code for evaluating the per-class performance of the two models,
and it prints a correct/incorrect image for three classes.

attack.ipybnb contains the code for generating the adversarial images based on the CNN model,
and it also handles analyzing the transferability to the MLP model and the pruned models.

pruning.ipynb contains the code for pruning and fine-tuning the CNN model, including
measuring the stats like latency and generating graphs.

models.py contains shared code for the models and loading datasets using transforms.

report.pdf is the written report for this project.

To load the checkpoints, run 'python3 check.py' in the command line.
Enter the path to the validation set, and then the index of the desired model.
After the accuracy is displayed, you can keep selecting models until you interrupt with Ctrl+C.