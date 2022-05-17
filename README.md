# Projet d'apprentissage faiblement supervisé
L'idée de ce projet est de classifier des images d'animaux selon 95 classes. 
## Apprentissage auto-supervisé avec SimCLR 
Nous allons nous inspirer de ce papier  
* Lien de l'article : https://arxiv.org/pdf/2002.05709.pdf?fbclid=IwAR3yQfp2O7dA1z164VpzO3hGOc453QJoVtvf0lrJQ__APkPS4Ch1LSMMuUw
* Github de l'article : https://github.com/google-research/simclr

Mais aussi de cet article de Stanford 
* https://hazyresearch.stanford.edu/blog/2022-04-19-contrastive-1

## Suivi de l'avancée du projet
* <del>create SIMCLR model</del>(done) 
* <del>contrastive loss </del> (done)
* <del> LARS optimizer </del> (done)
* <del> custom training loop</del> (done)
* <del> augmentations generator</del> (done)
* <del> entire training routine </del>(done)
* <del> contrastive loss problem (nan/inf values when temperature < 10)  --> need to correct this </del>(done)
* <del> projecteur last layer should be activation linear but only work when sigmoid --> need to correct this </del>(done)
* Bad results for the moment
* create checkpoints to save weights and history of training (to do)
* add augmentation to finetuning step (to do)
 
## Issues
*Large batch sizes do not improve model training
