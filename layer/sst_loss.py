import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config.config import config

class SSTLoss(nn.Module):
    def __init__(self, use_gpu=config['cuda']):
        super(SSTLoss, self).__init__()
        self.use_gpu = use_gpu
        self.max_object = config['max_object']
    def forward(self, input, target):

        batch_size = len(input)
        losses_pre = []
        losses_next = []
        losses_max = []
        losses_similarity = []
        losses_diff = []
        predicted_indexes = []
        for i in range(batch_size):
            input_pre = nn.Softmax(dim=3)(input[i])
            input_next = nn.Softmax(dim=2)(input[i])
            input_max = torch.max(input_pre, input_next)

            target_all = target[i].float()

            target_pre = target_all[:, :, :-1, :]
            target_next = target_all[:, :, :, :-1]
            target_max = target_all[:, :, :-1, :-1]
            target_pre_size = int(target_pre.sum().data[0])
            target_next_size = int(target_next.sum().data[0])
            target_max_size = int(target_max.sum().data[0])

            # loss_pre = torch.abs(target_pre - input_pre[:, :, :-1, :]).sum() / target_pre.size(2)
            # loss_next = torch.abs(target_next - input_next[:, :, :, :-1]).sum() / target_next.size(3)
            # loss_max = torch.abs(target_max - input_max[:, :, :-1, :-1]).sum() / target_pre.size(2)

            # loss_pre = -torch.log(torch.max(target_pre * input_pre[:, :, :-1, :] + 1e-6, 3)[0]).sum() / target_pre.size(2)
            # loss_next = -torch.log(torch.max(target_next*input_next[:, :, :, :-1] + 1e-6, 2)[0]).sum() / target_next.size(3)
            # loss_max = -torch.log(torch.max(target_max*input_max[:, :, :-1, :-1] + 1e-6, 3)[0]).sum() / target_pre.size(2)

            loss_pre = torch.abs((target_pre * torch.log(input_pre[:, :, :-1, :] + 1e-6))).sum() / target_pre.size(2)

            loss_next = torch.abs((target_next * torch.log(input_next[:, :, :, :-1] + 1e-6))).sum() / target_next.size(3)

            loss_max = torch.abs((target_max * torch.log(input_max[:, :, :-1, :-1]+1e-6))).sum() / target_pre.size(2)
            loss_diff = torch.abs(target_all-input_pre).sum() / input_pre.size(2)

            # loss_pre = - (target_pre * torch.log(input_pre[:, :, :-1, :])).sum() / (target_pre_size if target_pre_size > 0 else 1)
            #
            # loss_next = - (target_next * torch.log(input_next[:, :, :, :-1])).sum() / (target_next_size if target_next_size > 0 else 1)

            # loss_max = - (target_max * torch.log(input_max[:, :, :-1, :-1])).mean() / (target_max_size if target_max_size > 0 else 1)

            loss_similarity = (target_max * torch.abs(input_pre[:, :, :-1, :-1] - input_next[:, :, :-1, :-1])).mean()

            losses_pre.append(loss_pre)
            losses_next.append(loss_next)
            losses_max.append(loss_max)
            losses_similarity.append(loss_similarity)
            losses_diff.append(loss_diff)

            _, index_pre = input_pre[:, :, :-1, :].max(3)
            # print(input_pre.size(3))
            # print(index_pre)

            predicted_indexes.append(index_pre)

        return sum(losses_pre) / batch_size \
                , sum(losses_next) / batch_size \
                , sum(losses_similarity) / batch_size \
                , sum(losses_diff) / batch_size \
                , sum(losses_pre + losses_next + losses_max + losses_diff + losses_similarity) / (5*batch_size) \
                , predicted_indexes


        # mask_pre = mask0[:, :, :]
        # mask_next = mask1[:, :, :]
        # mask0 = mask0.unsqueeze(3).repeat(1, 1, 1, self.max_object+1)
        # mask1 = mask1.unsqueeze(2).repeat(1, 1, self.max_object+1, 1)
        # mask0 = Variable(mask0.data)
        # mask1 = Variable(mask1.data)
        # target = Variable(target.byte().data)
        #
        # if self.use_gpu:
        #     mask0 = mask0.cuda()
        #     mask1 = mask1.cuda()
        #
        # mask_region = (mask0 * mask1).float() # the valid position mask
        # mask_region_pre = mask_region.clone() #note: should use clone (fix this bug)
        # mask_region_pre[:, :, self.max_object, :] = 0
        # mask_region_next = mask_region.clone() #note: should use clone (fix this bug)
        # mask_region_next[:, :, :, self.max_object] = 0
        # mask_region_union = mask_region_pre*mask_region_next
        #
        # input_pre = nn.Softmax(dim=3)(mask_region_pre*input)
        # input_next = nn.Softmax(dim=2)(mask_region_next*input)
        # input_all = input_pre.clone()
        # input_all[:, :, :self.max_object, :self.max_object] = torch.max(input_pre, input_next)[:, :, :self.max_object, :self.max_object]
        # # input_all[:, :, :self.max_object, :self.max_object] = ((input_pre + input_next)/2.0)[:, :, :self.max_object, :self.max_object]
        # target = target.float()
        # target_pre = mask_region_pre * target
        # target_next = mask_region_next * target
        # target_union = mask_region_union * target
        # target_num = target.sum()
        # target_num_pre = target_pre.sum()
        # target_num_next = target_next.sum()
        # target_num_union = target_union.sum()
        # #todo: remove the last row negative effect
        # if int(target_num_pre.data[0]):
        #     loss_pre = - (target_pre * torch.log(input_pre)).sum() / target_num_pre
        # else:
        #     loss_pre = - (target_pre * torch.log(input_pre)).sum()
        # if int(target_num_next.data[0]):
        #     loss_next = - (target_next * torch.log(input_next)).sum() / target_num_next
        # else:
        #     loss_next = - (target_next * torch.log(input_next)).sum()
        # if int(target_num_pre.data[0]) and int(target_num_next.data[0]):
        #     loss = -(target_pre * torch.log(input_all)).sum() / target_num_pre
        # else:
        #     loss = -(target_pre * torch.log(input_all)).sum()
        #
        # if int(target_num_union.data[0]):
        #     loss_similarity = (target_union * (torch.abs((1-input_pre) - (1-input_next)))).sum() / target_num
        # else:
        #     loss_similarity = (target_union * (torch.abs((1-input_pre) - (1-input_next)))).sum()
        #
        # _, indexes_ = target_pre.max(3)
        # indexes_ = indexes_[:, :, :-1]
        # _, indexes_pre = input_all.max(3)
        # indexes_pre = indexes_pre[:, :, :-1]
        # mask_pre_num = mask_pre[:, :, :-1].sum().data[0]
        # if mask_pre_num:
        #     accuracy_pre = (indexes_pre[mask_pre[:, :, :-1]] == indexes_[mask_pre[:,:, :-1]]).float().sum() / mask_pre_num
        # else:
        #     accuracy_pre = (indexes_pre[mask_pre[:, :, :-1]] == indexes_[mask_pre[:, :, :-1]]).float().sum() + 1
        #
        # _, indexes_ = target_next.max(2)
        # indexes_ = indexes_[:, :, :-1]
        # _, indexes_next = input_next.max(2)
        # indexes_next = indexes_next[:, :, :-1]
        # mask_next_num = mask_next[:, :, :-1].sum().data[0]
        # if mask_next_num:
        #     accuracy_next = (indexes_next[mask_next[:, :, :-1]] == indexes_[mask_next[:, :, :-1]]).float().sum() / mask_next_num
        # else:
        #     accuracy_next = (indexes_next[mask_next[:, :, :-1]] == indexes_[mask_next[:, :, :-1]]).float().sum() + 1
        #
        # return loss_pre, loss_next, loss_similarity, \
        #        (loss_pre + loss_next + loss + loss_similarity)/4.0, accuracy_pre, accuracy_next, (accuracy_pre + accuracy_next)/2.0, indexes_pre

    def getProperty(self, input, target, mask0, mask1):
        return self.forward(input, target, mask0, mask1)
