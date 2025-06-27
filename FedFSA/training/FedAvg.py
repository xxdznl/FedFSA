import utils.utils
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import Dataset
from utils.utils import *
class DistributedTrainingDevice(object):

    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.loss = nn.CrossEntropyLoss()

    @torch.no_grad()
    def evaluate(self, data_x, data_y, dataset_name):
        self.model.eval()
        # testing
        loss_by_epoch_test = []
        accuracy_by_epoch_test = []

        n_tst = data_x.shape[0]
        tst_gen = DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=self.args.bs, shuffle=False)
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / self.args.bs))):
            data, target = tst_gen_iter.__next__()
            data, target = data.to(self.args.device), target.to(self.args.device)
            target = target.reshape(-1).long()
            output = self.model(data)
            total_loss_test = self.loss(output, target)

            prediction = torch.max(output, dim=1)[1]
            accuracy_test = torch.mean((prediction == target).float())

            loss_by_epoch_test.append(total_loss_test.item())
            accuracy_by_epoch_test.append(accuracy_test.item())

        loss_test = np.mean(loss_by_epoch_test)
        accuracy_test = np.mean(accuracy_by_epoch_test)
        accuracy_test = 100.00 * accuracy_test
        return accuracy_test, loss_test
        
class ClientFedAvg(DistributedTrainingDevice):

    def __init__(self, model, args, trn_x, trn_y, tst_x, tst_y, dataset_name, id_num=0):
        super().__init__(model, args)

        self.trn_gen = DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                                    batch_size=self.args.local_bs, shuffle=True)
        self.comm_vecs = {
            'local_update_list': None,
        }
        self.received_vecs = {
        }
        self.tst_x = tst_x
        self.tst_y = tst_y
        self.id = id_num
        self.max_norm = 10
        self.local_batch = int(np.ceil(trn_x.shape[0] / self.args.local_bs))
        self.lr = self.args.lr
        self.head = []
        self.body = []


    def seperate_param(self,model_name):# Get the last layer parameter name of the model: fc3 for Lenet and linear for Resnet18 
        if "CNNs" in model_name:
            self.head = [name for name, param in self.model.named_parameters() if 'fc3' in name]
        elif "ResNet18" in model_name:
            self.head = [name for name, param in self.model.named_parameters() if 'linear' in name]

        self.body = [name for name, param in self.model.named_parameters() if name not in self.head]

    def synchronize_with_server(self, server_comm_vecs):
        self.received_vecs = server_comm_vecs
        set_model_from_vector(self.args.device, self.model, self.received_vecs['Params_list'])

    def train_cnn(self,epoch, last):
        self.model.train()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, \
                        momentum=self.args.momentumForSGD,weight_decay=self.args.weight_decay)
        if last:
            for name,param in self.model.named_parameters():
                if name in self.body:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1)
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.args.lr_decay)

        # train and update
        epoch_loss = []
        train_accuracy = []
        for iter in range(self.args.local_ep):
            loss_by_epoch = []
            accuracy_by_epoch = []
            trn_gen_iter = self.trn_gen.__iter__()

            for i in range(self.local_batch):
                images, labels = trn_gen_iter.__next__()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                labels = labels.reshape(-1).long()
                output = self.model(images)
                total_loss = self.loss(output, labels)
                prediction = torch.max(output, dim=1)[1]
                accuracy = torch.mean((prediction == labels).float())
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)
                self.optimizer.step()
                loss_by_epoch.append(total_loss.item())
                accuracy_by_epoch.append(accuracy.item())
            self.scheduler.step()
            train_accuracy.append(sum(accuracy_by_epoch)/len(accuracy_by_epoch))
            epoch_loss.append(sum(loss_by_epoch) / len(loss_by_epoch))

        results_acc = sum(train_accuracy)/len(train_accuracy)
        results_loss = sum(epoch_loss) / len(epoch_loss)

        # client update decayed learning rate after local training
        if not last:
            self.lr = self.optimizer.param_groups[0]['lr']

        print('epoch:{}\tclient:[{}]\tlr:{}\ttrain_loss=\t{:.5f}\tacc_train=\t{:.5f}'.
                            format(epoch,self.id,self.lr,results_loss, results_acc))            
        # get local model parameter of vector form
        last_state_params_list = get_mdl_params(self.model)
        # local model parameter updates
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']

class ServerFedAvg(DistributedTrainingDevice):

    def __init__(self, model, args, init_par_list):
        super().__init__(model, args)
        self.server_model_params_list = init_par_list
        self.clients_params_list = init_par_list.repeat(args.num_users, 1)
        self.clients_updated_params_list = torch.zeros((args.num_users, init_par_list.shape[0])) 
        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
        }
        self.Averaged_update = torch.zeros(self.server_model_params_list.shape)

    def global_update(self, participating_clients_ids):
        self.Averaged_update = torch.mean(self.clients_updated_params_list[participating_clients_ids], dim=0)
        # self.Averaged_model  = torch.mean(self.clients_params_list[participating_clients_ids], dim=0)
        self.server_model_params_list = self.server_model_params_list + self.args.globallr * self.Averaged_update
        set_model_from_vector(self.args.device, self.model, self.server_model_params_list)

    # Each client will call this after training
    def receive_from_clienti(self, client_id,client_comm_vecs):
        self.clients_updated_params_list[client_id] = client_comm_vecs['local_update_list']
    # Each client will call this before training
    def process_for_communication(self, client_id):
        if not self.args.use_RI:
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list)
        else:
            # RI adopts the w(i,t) = w(t) + beta[w(t) - w(i,K,t-1)] as initialization
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list + self.args.beta\
                                    * (self.server_model_params_list - self.clients_params_list[client_id]))