import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--adaptive', action='store_true', default=False, help="adaptively tune rho of SAM")
    parser.add_argument('--epochs', type=int, default=500, help="rounds of training")
    parser.add_argument('--model', type=str, default='CNNs', help='whether CNNs or ResNet18')
    parser.add_argument('--test_freq', type=int, default=1, help="frequency of test")
    parser.add_argument('--num_users', type=int, default=100, help="number of clients")
    parser.add_argument('--frac', type=float, default=0.1, help="the sampling ratio of clients")
    parser.add_argument('--local_ep', type=int, default=2, help="the number of local epochs")
    parser.add_argument('--last_local_ep', type=int, default=10, help="the number of local epochs of last")
    parser.add_argument('--local_bs', type=int, default=48, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=10, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="local learning rate")
    parser.add_argument('--globallr', type=float, default=1, help="global learning rate")
    parser.add_argument('--momentumForSGD', type=float, default=0, help="local SGD momentum")
    parser.add_argument('--weight_decay', type=float, default=0, help="local SGD weight_decay")
    parser.add_argument('--lr_decay', type=float, default=0.997, help='the value of local learning rate decay')
    parser.add_argument('--beta', default=0.1, type=float, help='the coefficient for relaxed initialization') 
    parser.add_argument('--sync', type=str, default='True', help='If the client is synchronized with server')
    parser.add_argument('--method', type=str, default='FedAvg', help='method name')
    # SAM_based methods arguments
    parser.add_argument('--rho_default',default = 0.1,type=float,help='default perturbation amplitude for SAM_based FL methods')
    parser.add_argument('--alpha', default=0, type=float, help='local SAM momentum')
    # FSA arguments
    parser.add_argument('--rho_larger',default = 0.1,type=float,help='larger perturbation amplitude for FSA')
    parser.add_argument('--TopC',default = 2,type=int,help='TopC layer to employ larger perturbation')
    # FedSpeed FedSMOO arguments
    parser.add_argument('--use-RI', action='store_true', default=False)  # activate if use relaxed initialization (RI)
    parser.add_argument('--lamb', default=0.1, type=float)               # select the coefficient for the prox-term

    # noniid data partioning strategy
    parser.add_argument('--dataset', type=str, default='CIFAR10', help="name of dataset")
    parser.add_argument('--frac_data', type=float, default=0.7, help="training set ratio of client's data")
    parser.add_argument('--rule', type=str, default='noniid', help='whether noniid or Dirichet')
    parser.add_argument('--class_main', type=int, default=5, help='the value of class_main for noniid')
    parser.add_argument('--dir_a', default=0.5, type=float, help='the value of dir_a for dirichlet')
    #ALA
    parser.add_argument('--s', type=int, default=80, help="how many datas for ALA")
    parser.add_argument('--p', type=int, default=2, help="how many layers for ALA")
    
    # other arguments
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=23, help='random seed')
    parser.add_argument('--filepath', type=str, default='filepath', help='results saved in')
    args = parser.parse_args()
    return args


