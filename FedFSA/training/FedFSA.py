import torch
from torch import nn
import numpy as np
from samOptimizers.FSA import FSA
from utils.utils import *
from training.FedAvg import ClientFedAvg,ServerFedAvg
class ClientFedFSA(ClientFedAvg):

    def __init__(self, model, args, trn_x, trn_y, tst_x, tst_y, dataset_name, id_num):
        super().__init__(model, args, trn_x, trn_y, tst_x, tst_y, dataset_name, id_num)
        self.rho_default = self.args.rho_default
        self.rho_larger = self.args.rho_larger
        self.TopC = self.args.TopC
        self.max2Layer = []

    def train_cnn(self,epoch, last):
        self.model.train()
        params_list = [param.clone().detach() for param in self.model.parameters()]
        self.base_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, \
                        momentum=self.args.momentumForSGD,weight_decay=self.args.weight_decay)
        if last:
            for name,param in self.model.named_parameters():
                if name in self.body:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.base_optimizer, step_size=1, gamma=1)
        else:
            self.optimizer = FSA(self.model.parameters(),self.base_optimizer,cid=self.id,rho_default=self.rho_default,rho_larger=self.rho_larger)
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
                if not last:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)
                    self.optimizer.first_step(max2Layer=self.max2Layer)
                    output = self.model(images)
                    total_loss2 = self.args.alpha * self.loss(output, labels)
                    self.optimizer.zero_grad()
                    total_loss2.backward()
                    self.optimizer.second_step()
                    param_list = param_to_vector(self.model)
                    delta_list = self.received_vecs['Client_momentum'].to(self.args.device)
                    loss_correct = (1 - self.args.alpha) * torch.sum(param_list * delta_list)
                    loss_correct.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)
                    self.base_optimizer.step()
                else:
                    self.base_optimizer.zero_grad()
                    total_loss.backward()
                    self.base_optimizer.step()
                loss_by_epoch.append(total_loss.item())
                accuracy_by_epoch.append(accuracy.item())
            self.scheduler.step()
            train_accuracy.append(sum(accuracy_by_epoch)/len(accuracy_by_epoch))
            epoch_loss.append(sum(loss_by_epoch) / len(loss_by_epoch))
        if not last:
            ############### uncomment code below to switch to Random, more details please see ablation study in our paper#########################
            
            # random for CNNs : number of weights layer
            # numbers = [0, 2, 4, 6, 8]
            # self.max2Layer = random.sample(numbers, 2)

            # random for ResNet18 : number of weights layer
            # numbers = [0, 3, 6, 9,12,15,18,24,27,30,33,39,42,45,48,54,57,60]
            # self.max2Layer = random.sample(numbers, 5)

            ############### uncomment code above to switch to Random #########################

            ############### uncomment code below to switch to FSA #########################
            self.max2Layer = self.optimizer.nextDisruptLayer(params_list,self.model.named_parameters(),TopC=self.TopC)
            ############### uncomment code above to switch to FSA #########################
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

'''
------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------
'''


class ServerFedFSA(ServerFedAvg):

    def __init__(self, model, args, init_par_list,shape_data):
        super().__init__(model, args,init_par_list)
        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
            'Client_momentum': torch.zeros((init_par_list.shape[0])),
        }
        self.local_iteration = self.args.local_ep * int(np.ceil(shape_data / self.args.local_bs))

    def process_for_communication(self, client_id):
        super().process_for_communication(client_id)
        # last lr = current lr/lr_decay
        self.comm_vecs['Client_momentum'].copy_(self.Averaged_update / self.local_iteration / self.args.lr * self.args.lr_decay * -1.)


