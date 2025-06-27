import ssl
import copy
from utils.utils import *
from utils.options import args_parser
from nets.CNNs import client_model
from nets.ResNet import ResNet18
from utils.dataset import DatasetObject
from training.FedAvg import ClientFedAvg, ServerFedAvg
from training.FedALA import ClientFedALA, ServerFedALA
from training.FedFSA import ClientFedFSA, ServerFedFSA
from training.MoFedSAM import ClientMoFedSAM, ServerMoFedSAM
from training.FedSpeed import ClientFedSpeed, ServerFedSpeed
from training.FedSMOO import ClientFedSMOO, ServerFedSMOO
torch.set_printoptions(
    precision=8,
    threshold=10,
    edgeitems=1,
    linewidth=100,
    profile=None,
    sci_mode=False
)
np.set_printoptions(suppress=True, formatter={'float': '{:.8f}'.format})
# Define the mapping between methods and clients and servers
method_map = {
    'FedSpeed': (ClientFedSpeed, ServerFedSpeed),
    'FedSMOO': (ClientFedSMOO, ServerFedSMOO),
    'FedFSA': (ClientFedFSA, ServerFedFSA),
    'FedALA': (ClientFedALA, ServerFedALA),
    'MoFedSAM': (ClientMoFedSAM, ServerMoFedSAM),
    'FedAvg': (ClientFedAvg, ServerFedAvg)
}
# Define the mapping between models and datasets
dataset_model_map = {
    'CIFAR100': {
        'CNNs': 'cifar100_CNNs',
        'ResNet18': 100
    },
    'FMNIST': {
        'CNNs': 'FMNIST_CNNs',
        'ResNet18': 10
    },
    'CIFAR10': {
        'CNNs': 'cifar10_CNNs',
        'ResNet18': 10
    },
    'TINY': {
        'CNNs': 'tiny_CNNs',
        'ResNet18': 200
    }
}
if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    # parse args
    args = args_parser()
    setup_seed(args.seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    data_obj = DatasetObject(dataset=args.dataset, n_client=args.num_users, seed=args.seed, rule=args.rule,
                            class_main=args.class_main, frac_data=args.frac_data, dir_alpha=args.dir_a)

    # Get the client and server of corresponding method
    if args.method in method_map:
        Client, Server = method_map[args.method]
    else:
        exit('Error: unrecognized method')
    # Check the validity of the dataset and model
    if args.dataset not in dataset_model_map:
        exit('Error: unrecognized dataset')
    if args.model not in dataset_model_map[args.dataset]:
        exit('Error: unrecognized model')
    # Get the corresponding model
    model_info = dataset_model_map[args.dataset][args.model]
    if args.model == 'CNNs':
        net_glob = client_model(model_info).to(args.device)
    elif args.model == 'ResNet18':
        net_glob = ResNet18(model_info)
    # Initialize the client and server
    clients = [Client(model=copy.deepcopy(net_glob).to(args.device), args=args, trn_x=data_obj.clnt_x[i],
                    trn_y=data_obj.clnt_y[i], tst_x=data_obj.tst_x[i], tst_y=data_obj.tst_y[i],
                    dataset_name=data_obj.dataset, id_num=i) for i in range(args.num_users)]
    if args.method != 'FedAvg':
        server = Server(model=(net_glob).to(args.device), args=args, init_par_list=get_mdl_params(net_glob),
                    shape_data=data_obj.clnt_x[0].shape[0])
    else:
        server = Server(model=(net_glob).to(args.device), args=args, init_par_list=get_mdl_params(net_glob))
    W = {name: value for name, value in net_glob.named_parameters()}
    total_num_layers = len(W)
    net_keys = [*W.keys()]
    logger = get_logger(args.filepath)
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('%s: %s' % (k, vars(args)[k]))
    logger.info('--------args----------\n')
    logger.info('total_num_layers')
    logger.info(total_num_layers)
    logger.info('net_keys')
    logger.info(net_keys)
    logger.info('start training!')
    # save max_accuracy
    max_results_acc = -float('inf')
    max_results_acc_glob = -float('inf')
    for client in clients:
        client.seperate_param(net_glob.name)
    for epoch_i in range(args.epochs + 1):
        # net_glob.train()
        m = max(int(args.frac * args.num_users), 1)
        if epoch_i == args.epochs:
            m = args.num_users
        participating_clients = random.sample(clients, m)
        participating_clients_ids = [client.id for client in participating_clients]
        last = epoch_i == args.epochs

        for client in participating_clients:
            server.process_for_communication(client.id)
            if args.sync == 'True':
                client.synchronize_with_server(server.comm_vecs)
            client.train_cnn(epoch_i, last)
            server.receive_from_clienti(client.id, client.comm_vecs)
        server.global_update(participating_clients_ids)

        # -----------------------------------------------test--------------------------------------------------------------------

        # -----------------------------------------------test--------------------------------------------------------------------

        if epoch_i % args.test_freq == args.test_freq - 1 or epoch_i >= args.epochs - 10:
            results_loss = []; results_acc = []
            results_loss_glob = []; results_acc_glob = []
            for client in clients:
                results_test, loss_test = client.evaluate(data_x=client.tst_x, data_y=client.tst_y,
                                                        dataset_name=data_obj.dataset)
                results_test2, loss_test2 = server.evaluate(data_x=client.tst_x, data_y=client.tst_y,
                                                        dataset_name=data_obj.dataset)
                results_loss.append(loss_test)
                results_acc.append(results_test)
                results_loss_glob.append(loss_test2)
                results_acc_glob.append(results_test2)
            results_loss = np.mean(results_loss)
            results_acc = np.mean(results_acc)
            results_loss_glob = np.mean(results_loss_glob)
            results_acc_glob = np.mean(results_acc_glob)
            max_results_acc = max(max_results_acc, results_acc)
            max_results_acc_glob = max(max_results_acc_glob, results_acc_glob)

            if last:
                logger.info('*************************************************************************')
                logger.info('Final Epoch:[{}]\tloss_glob=\t{:.5f}\tacc_test_glob=\t{:.5f}\t max_acc_test_glob=\t{:.5f}'.
                            format(epoch_i, results_loss_glob, results_acc_glob, max_results_acc_glob))
                logger.info('Final Epoch:[{}]\tloss=\t{:.5f}\tacc_test=\t{:.5f}\tmax_acc_test=\t{:.5f}'.
                            format(epoch_i, results_loss, results_acc, max_results_acc))
                logger.info('*************************************************************************')
            else:
                logger.info('*************************************************************************')
                logger.info('Epoch:[{}]\tloss_glob=\t{:.5f}\tacc_test_glob=\t{:.5f}'.
                            format(epoch_i, results_loss_glob, results_acc_glob))
                logger.info('Epoch:[{}]\tloss=\t{:.5f}\tacc_test=\t{:.5f}'.
                            format(epoch_i, results_loss, results_acc))
                logger.info('*************************************************************************')

    logger.info('finish training!')





