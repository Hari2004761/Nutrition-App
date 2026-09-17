Previous 20-class run, kept for comparison.

Trained before the data-split fix: Food-101's test split was used as the
validation set during training AND as the final evaluation set, so the
checkpoint was selected on the same images it was graded on. Reported
val/test accuracy: 90.7%.

classifier_best.pt  - the checkpoint that run produced
history.csv         - a copy of outputs/history.csv as it stood at the backup
                      (the live file keeps these rows; the new run is appended)
