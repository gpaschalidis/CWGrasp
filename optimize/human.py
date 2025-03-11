#
# Copyright (C) 2022 Universiteit van Amsterdam (UvA).
# All rights reserved.
#
# Universiteit van Amsterdam (UvA) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with UvA or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: g.paschalidis@uva.nl
#

import torch
import smplx
import numpy as np

from .losses.loss import Losses

class Human(torch.nn.Module):
    def __init__(
            self, 
            body_params, 
            gender,
            grasp_type, 
            smplx_path, 
            num_samples,
            hand_verts, 
            rec_list,
            obj_transl,
            config, 
            device):
        super(Human, self).__init__()   
        self.body_params = body_params
        self.grasp_type = grasp_type
        self.hverts = hand_verts
        self.initial_bpose = body_params["body_pose"].clone()
        self.body_pose = None
        self.transl_x = None
        self.transl_y = None
        self.transl_z = None
        self.global_orient_x = None
        self.global_orient_y = None
        self.global_orient_z = None
        if grasp_type == "right":
            self.int_shoulder_old = self.initial_bpose[:,48::51]
            self.int_elbow_old = self.initial_bpose[:,54:57]
            self.int_wrist_old = self.initial_bpose[:,60:63]
        elif grasp_type == "left":
            self.int_shoulder_old = self.initial_bpose[:,45:48]
            self.int_elbow_old = self.initial_bpose[:,51:54]
            self.int_wrist_old = self.initial_bpose[:,57:60]
        self.int_wrist_pose = None
        self.int_elbow_pose = None
        self.int_shoulder_pose = None
        self.factor = 50
        self.wrist_weight = 10
        self.sbj_m = smplx.create(model_path=smplx_path,
                        model_type='smplx',
                        gender=gender,
                        use_pca=False,
                        num_pca_comps=np.array(45),
                        flat_hand_mean=True,
                        batch_size=num_samples).to(device).eval()

        self.loss = Losses(config, grasp_type, device)
        
        self.faces = torch.tensor(
            self.sbj_m.faces.astype("int64")
        ).to(device)

        self.human_vec = None
        self.obstacle_list = rec_list
        self.obj_transl = torch.tensor(obj_transl).to(torch.float32).to(device)


    def forward(self):
        self.body_params["body_pose"] = self.body_pose
        self.body_params["transl"] = torch.cat((
                self.transl_x.unsqueeze(1), 
                self.transl_y.unsqueeze(1), 
                self.transl_z.unsqueeze(1)),1)
        self.body_params["global_orient"] = torch.cat((
                self.global_orient_x.unsqueeze(1), 
                self.global_orient_y.unsqueeze(1), 
                self.global_orient_z.unsqueeze(1)),1)
        bm = self.sbj_m(**self.body_params)

        body_verts = bm.vertices
        pred_pelvis_joint = bm.joints[:, 0, :]
        mean_foot = (bm.joints[:,8] + bm.joints[:,7])/2
        pred_human_vec = pred_pelvis_joint - mean_foot
        
        pred_human_vec = pred_human_vec /(((pred_human_vec**2).sum(1)).sqrt()[...,None] +  1e-10)

        total_loss_dict = self.loss.total_loss(
                body_verts,
                self.hverts,
                self.faces,
                self.obstacle_list,
                self.body_pose,
                self.initial_bpose,
                self.human_vec,
                pred_human_vec,
                self.obj_transl,
                self.wrist_weight
                )
        return total_loss_dict

    def final_step(self):
        if self.grasp_type == "right":
            self.body_params["body_pose"] = torch.cat((
                self.body_params["body_pose"][:,0:48],
                self.int_shoulder_pose,
                self.body_params["body_pose"][:,51:54],
                self.int_elbow_pose,
                self.body_params["body_pose"][:,57:60],
                self.int_wrist_pose),1)
        elif self.grasp_type == "left":
            self.body_params["body_pose"] = torch.cat((
                self.body_params["body_pose"][:,0:45],
                self.int_shoulder_pose,
                self.body_params["body_pose"][:,48:51],
                self.int_elbow_pose,
                self.body_params["body_pose"][:,54:57],
                self.int_wrist_pose,
                self.body_params["body_pose"][:,60:63]),1)

        bm = self.sbj_m(**self.body_params)

        body_verts = bm.vertices

        total_loss_dict = self.loss.final_loss(
            body_verts,
            self.hverts,
            self.faces,
            self.factor,
            self.body_pose,
            self.initial_bpose,
            self.obj_transl,
            self.int_shoulder_pose,
            self.int_shoulder_old,
            self.int_elbow_pose,
            self.int_elbow_old,
            self.int_wrist_pose,
            self.int_wrist_old
        )
        return total_loss_dict
        

    def set_opt_parameters(self):
        self.body_pose = torch.nn.Parameter(self.body_params["body_pose"], requires_grad=True)
        self.transl_x = torch.nn.Parameter(self.body_params["transl"][:,0], requires_grad=True)
        self.transl_y = torch.nn.Parameter(self.body_params["transl"][:,1], requires_grad=True)
        self.transl_z = torch.nn.Parameter(self.body_params["transl"][:,2], requires_grad=True)
        self.global_orient_x = torch.nn.Parameter(self.body_params["global_orient"][:,0], requires_grad=True)
        self.global_orient_y = torch.nn.Parameter(self.body_params["global_orient"][:,1], requires_grad=True)
        self.global_orient_z = torch.nn.Parameter(self.body_params["global_orient"][:,2], requires_grad=True)
    
    def reset_opt_parameters(self):
        self.body_pose.requires_grad = False
        self.transl_x.requires_grad = False
        self.transl_y.requires_grad = False
        self.transl_z.requires_grad = False
        self.global_orient_x.requires_grad = False
        self.global_orient_y.requires_grad =False
        self.global_orient_z.requires_grad =False
        if self.grasp_type == "right":
            self.int_wrist_pose = torch.nn.Parameter(self.body_params["body_pose"][:,60:63].detach(), requires_grad=True)
            self.int_elbow_pose = torch.nn.Parameter(self.body_params["body_pose"][:,54:57].detach(), requires_grad=True)
            self.int_shoulder_pose = torch.nn.Parameter(self.body_params["body_pose"][:,48:51].detach(), requires_grad=True)
        elif self.grasp_type == "left":
            self.int_wrist_pose = torch.nn.Parameter(self.body_params["body_pose"][:,57:60].detach(), requires_grad=True)
            self.int_elbow_pose = torch.nn.Parameter(self.body_params["body_pose"][:,51:54].detach(), requires_grad=True)
            self.int_shoulder_pose = torch.nn.Parameter(self.body_params["body_pose"][:,45:48].detach(), requires_grad=True)
 
