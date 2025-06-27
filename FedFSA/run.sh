## examples ###############
#FedALA + FSA
nohup bash -c "python ./main.py --filepath ./result/CIFAR100ALA_withFSA.txt --dataset CIFAR100 --method FedALA --model CNNs --num_users 100 --frac 0.1 --lr 0.1 --local_ep 10 --lr_decay 1 --rule Dirichlet --dir_a 0.3 --gpu 0 --epoch 500 --bs 128 --local_bs 48 --seed 23 --class_main 15 --lamb 0.1 --alpha 0.1 --rho_default 0.05 --rho_larger 0.9 --TopC 2" > ./result/CIFAR100ALA_withFSA_nohup.txt  2>&1 &
#FedSpeed
nohup bash -c "python ./main.py --filepath ./result/CIFAR100Speed.txt --dataset CIFAR100 --method FedSpeed --model CNNs --num_users 100 --frac 0.1 --lr 0.1 --local_ep 10 --lr_decay 0.998 --rule Dirichlet --dir_a 0.3 --gpu 0 --epoch 500 --bs 128 --local_bs 48 --seed 23 --class_main 15 --lamb 0.1 --rho_default 0.1 --rho_larger 0.1 --alpha 0.1" > ./result/CIFAR100Speed_nohup.txt 2>&1 &
#FedSMOO
nohup bash -c "python ./main.py --filepath ./result/CIFAR100SMOO.txt --dataset CIFAR100 --method FedSMOO --model CNNs --num_users 100 --frac 0.1 --lr 0.1 --local_ep 10 --lr_decay 0.998 --rule Dirichlet --dir_a 0.3 --gpu 1 --epoch 500 --bs 128 --local_bs 48 --seed 23 --class_main 15 --lamb 0.1 --rho_default 0.1 --rho_larger 0.1 --alpha 0.1" > ./result/CIFAR100SMOO_nohup.txt 2>&1 &
#MoFedSAM
nohup bash -c "python ./main.py --filepath ./result/CIFAR100MO.txt --dataset CIFAR100 --method MoFedSAM --model CNNs --num_users 100 --frac 0.1 --lr 0.1 --local_ep 10 --lr_decay 1 --rule Dirichlet --dir_a 0.3 --gpu 0 --epoch 500 --bs 128 --local_bs 48 --seed 23 --class_main 15 --lamb 0.1 --rho_default 0.1 --rho_larger 0.1 --TopC 2 --alpha 0.1" > ./result/CIFAR100MO_nohup.txt 2>&1 &
#FedFSA
nohup bash -c "python ./main.py --filepath ./result/CIFAR100FSA.txt --dataset CIFAR100 --method FedFSA --model CNNs --num_users 100 --frac 0.1 --lr 0.1 --local_ep 10 --lr_decay 1 --rule Dirichlet --dir_a 0.3 --gpu 3 --epoch 500 --bs 128 --local_bs 48 --seed 23 --class_main 15 --lamb 0.1 --rho_default 0.05 --rho_larger 0.9 --TopC 2 --alpha 0.1" > ./result/CIFAR100FSA_nohup.txt 2>&1 &

#FedAVG resnet
nohup bash -c "python ./main.py --filepath ./result/CIFAR100Avg_ResNet18.txt --dataset CIFAR100 --method FedAvg --model ResNet18 --num_users 100 --frac 0.1 --lr 0.1 --local_ep 10 --lr_decay 1 --rule Dirichlet --dir_a 0.3 --gpu 0 --epoch 500 --bs 128 --local_bs 48 --seed 23 --class_main 15 --lamb 0.1 --rho_default 0.1 --rho_larger 0.1 --alpha 0.1" > ./result/nohupres/CIFAR100Avg_ResNet18_nohup.txt 2>&1 &
#FedSMOO resnet
nohup bash -c "python ./main.py --filepath ./result/CIFAR100SMOO_ResNet18.txt --dataset CIFAR100 --method FedSMOO --model ResNet18 --num_users 100 --frac 0.1 --lr 0.1 --local_ep 10 --lr_decay 0.998 --rule Dirichlet --dir_a 0.3 --gpu 0 --epoch 500 --bs 128 --local_bs 48 --seed 23 --class_main 15 --lamb 0.1 --rho_default 0.1 --rho_larger 0.1 --alpha 0.1" > ./result/nohupres/CIFAR100SMOO_ResNet18_nohup.txt 2>&1 &
#FedSpeed resnet
nohup bash -c "python ./main.py --filepath ./result/CIFAR100Speed_ResNet18.txt --dataset CIFAR100 --method FedSpeed --model ResNet18 --num_users 100 --frac 0.1 --lr 0.1 --local_ep 10 --lr_decay 0.998 --rule Dirichlet --dir_a 0.3 --gpu 0 --epoch 500 --bs 128 --local_bs 48 --seed 23 --class_main 15 --lamb 0.1 --rho_default 0.1 --rho_larger 0.1 --alpha 0.1" > ./result/nohupres/CIFAR100Speed_ResNet18_nohup.txt 2>&1 &
## examples ###############