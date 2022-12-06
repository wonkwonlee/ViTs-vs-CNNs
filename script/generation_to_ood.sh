# ResNet-18 Aligned Version
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet18' --test-batch 256  --data /scratch/wl2733/cv/data/imagenet-a  --evaluate --imagenet-a--ckpt /scratch/wl2733/cv/ckpt/res18_align.pth  --sing singgbn

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet18' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-stylized --evaluate--ckpt /scratch/wl2733/cv/ckpt/res18_align.pth  --sing singgbn

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet18' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-c --evaluate_imagenet_c--ckpt /scratch/wl2733/cv/ckpt/res18_align.pth --sing singgbn



# ResNet-18 Best Version
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet18' --test-batch 256  --data /scratch/wl2733/cv/data/imagenet-a  --evaluate --imagenet-a--ckpt /scratch/wl2733/cv/ckpt/res18_best.pth  --sing singgbn

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet18' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-stylized --evaluate--ckpt /scratch/wl2733/cv/ckpt/res18_best.pth  --sing singgbn

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet18' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-c --evaluate_imagenet_c--ckpt /scratch/wl2733/cv/ckpt/res18_best.pth --sing singgbn




# ResNet-50 Aligned Version
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet50' --test-batch 256  --data /scratch/wl2733/cv/data/imagenet-a  --evaluate --imagenet-a--ckpt /scratch/wl2733/cv/ckpt/res50_aligned.pth  --sing singbn

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet50' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-stylized --evaluate--ckpt /scratch/wl2733/cv/ckpt/res50_aligned.pth  --sing singbn

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet50' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-c --evaluate_imagenet_c--ckpt /scratch/wl2733/cv/ckpt/res50_aligned.pth --sing singbn


# ResNet-50 Best Version
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet50' --test-batch 256  --data /scratch/wl2733/cv/data/imagenet-a  --evaluate --imagenet-a--ckpt /scratch/wl2733/cv/ckpt/res50_best.pth  --sing singgbn

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet50' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-stylized --evaluate--ckpt /scratch/wl2733/cv/ckpt/res50_best.pth  --sing singgbn

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet50' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-c --evaluate_imagenet_c--ckpt /scratch/wl2733/cv/ckpt/res50_best.pth --sing singgbn



# ResNet-50 Original Version
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet50' --test-batch 256  --data /scratch/wl2733/cv/data/imagenet-a  --evaluate --imagenet-a--ckpt /scratch/wl2733/cv/ckpt/res50_ori.pth  --sing singbn

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet50' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-stylized --evaluate--ckpt /scratch/wl2733/cv/ckpt/res50_ori.pth  --sing singbn

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet50' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-c --evaluate_imagenet_c--ckpt /scratch/wl2733/cv/ckpt/res50_ori.pth --sing singbn





# ResNet-101 Aligned Version
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet101' --test-batch 256  --data /scratch/wl2733/cv/data/imagenet-a  --evaluate --imagenet-a--ckpt /scratch/wl2733/cv/ckpt/res101_align.pth  --sing singbn

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet101' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-stylized --evaluate--ckpt /scratch/wl2733/cv/ckpt/res101_align.pth  --sing singbn

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet101' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-c --evaluate_imagenet_c--ckpt /scratch/wl2733/cv/ckpt/res101_align.pth --sing singbn




# ResNet-101 Best Version
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet101' --test-batch 256  --data /scratch/wl2733/cv/data/imagenet-a  --evaluate --imagenet-a--ckpt /scratch/wl2733/cv/ckpt/res101_best.pth  --sing singgbn

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet101' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-stylized --evaluate--ckpt /scratch/wl2733/cv/ckpt/res101_best.pth  --sing singgbn

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet101' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-c --evaluate_imagenet_c--ckpt /scratch/wl2733/cv/ckpt/res101_best.pth --sing singgbn




# DeiT-Mini.
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'deit_mini_patch16_224' --test-batch 256  --data /scratch/wl2733/cv/data/imagenet-a  --evaluate --imagenet-a--ckpt /scratch/wl2733/cv/ckpt/deitmini.pth #--sing singln

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'deit_mini_patch16_224' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-stylized --evaluate--ckpt /scratch/wl2733/cv/ckpt/deitmini.pth  #--sing singln

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'deit_mini_patch16_224' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-c --evaluate_imagenet_c--ckpt /scratch/wl2733/cv/ckpt/deitmini.pth #--sing singln




# DeiT-Small Distillation Model Evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'deit_small_patch16_224' --test-batch 256  --data /scratch/wl2733/cv/data/imagenet-a  --evaluate --imagenet-a--ckpt /scratch/wl2733/cv/ckpt/deitsmall.pth --sing singln

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'deit_small_patch16_224' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-stylized --evaluate--ckpt /scratch/wl2733/cv/ckpt/deitsmall.pth  --sing singln

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'deit_small_patch16_224' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-c --evaluate_imagenet_c--ckpt /scratch/wl2733/cv/ckpt/deitsmall.pth --sing singln




# Deit-Base-300 epoch, checkpoint can be download from DeiT repo.
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'deit_base_patch16_224' --test-batch 256  --data /scratch/wl2733/cv/data/imagenet-a  --evaluate --imagenet-a--ckpt /scratch/wl2733/cv/ckpt/deit_base_patch16_224-b5f2ef4d.pth --sing singbn

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'deit_base_patch16_224' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-stylized --evaluate--ckpt /scratch/wl2733/cv/ckpt/deit_base_patch16_224-b5f2ef4d.pth  --sing singbn

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'deit_base_patch16_224' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-c --evaluate_imagenet_c--ckpt /scratch/wl2733/cv/ckpt/deit_base_patch16_224-b5f2ef4d.pth --sing singbn





# DeiT Knowledge Distillation Model Evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'deit_small_patch16_224' --test-batch 256  --data /scratch/wl2733/cv/data/imagenet-a  --evaluate --imagenet-a--ckpt /scratch/wl2733/cv/ckpt/deits_distil.pth --sing singln

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'deit_small_patch16_224' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-stylized --evaluate--ckpt /scratch/wl2733/cv/ckpt/deits_distil.pth  --sing singln

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'deit_small_patch16_224' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-c --evaluate_imagenet_c--ckpt /scratch/wl2733/cv/ckpt/deits_distil.pth --sing singln



# ResNet Knowledge Distillation Model Evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet50' --test-batch 256  --data /scratch/wl2733/cv/data/imagenet-a  --evaluate --imagenet-a--ckpt /scratch/wl2733/cv/ckpt/res50_distill.pth --sing singgbn

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet50' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-stylized --evaluate--ckpt /scratch/wl2733/cv/ckpt/res50_distill.pth  --sing singgbn

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_robust.py -a 'resnet50' --test-batch 256 --data /scratch/wl2733/cv/data/imagenet-c --evaluate_imagenet_c--ckpt /scratch/wl2733/cv/ckpt/res50_distill.pth --sing singgbn

