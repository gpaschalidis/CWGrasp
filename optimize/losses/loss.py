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


import numpy as np
import torch
import kaolin
import open3d as o3d
import scipy.sparse as sp
import pickle as pkl
from collections import Counter
from ..mesh_utils import HDfier



def create_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(list(np.random.rand(1,3)[0]))
    return pcd


def create_mesh(verts, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.paint_uniform_color(list(np.random.rand(1,3)[0]))
    mesh.compute_vertex_normals()
    return mesh


def create_o3d_box_mesh(rec_verts):
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=3,height=3,depth=0.005)
    mesh_box_verts = np.array(mesh_box.vertices)
    mesh_box_center = (mesh_box_verts.max(0)+mesh_box_verts.min(0))/2
    mesh_box_verts -= mesh_box_center
    mesh_box_verts[:,0] += rec_verts.mean(0)[0]
    mesh_box_verts[:,1] += rec_verts.mean(0)[1]
    mesh_box.vertices = o3d.utility.Vector3dVector(mesh_box_verts)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.2, 0.2, 0.2])
    return mesh_box


def to_scipy_sparse_matrix(
    edge_index,
    num_nodes=None,
) -> sp.coo_matrix:

    row, col = edge_index
    edge_attr = torch.ones(row.shape[0])
    out = sp.coo_matrix((edge_attr, (row, col)), (num_nodes, num_nodes))
    return out


class Losses(object):
    def __init__(self, config, grasp_type, device):
        # Preliminaries.
        self.config = config
        self.device = device
        self.subsample_sbj = config["subsample_sbj"] 
        self.sbj_verts_region_map = np.load(config["sbj_verts_region_map_pth"], allow_pickle=True)  # (10475,)
        if config["subsample_sbj"]:
            self.sbj_verts_id = np.load(config["sbj_verts_simplified"])                             # (625,)
            self.sbj_faces_simplified = torch.tensor(np.load(config["sbj_faces_simplified"])).to(device)
            self.sbj_verts_region_map = self.sbj_verts_region_map[self.sbj_verts_id]
            self.adj_matrix_simplified = np.load(config["adj_matrix_simplified"])
        with open(config["mano2smplx_verts_ids_both"],"rb") as f : 
            correspondences = pkl.load(f)
        
        self.correspondences = correspondences["{}_hand".format(grasp_type)]

        self.interesting = dict(np.load(config["interesting_pth"]))                                    
        self.hdfy_op = HDfier(model_type="smplx")
        self.b1 = 1
        self.b2 = 0.15

    def intersection(self, sbj_verts, obj_verts, sbj_faces, obj_faces, full_body=True, adjacency_matrix=None):
        device = sbj_verts.device
        bs = sbj_verts.shape[0]
        obj_verts = obj_verts.repeat(bs, 1, 1)                                                                               # (bs, N_obj, 3)
        num_obj_verts, num_sbj_verts = obj_verts.shape[1], sbj_verts.shape[1]
        penet_loss_batched_in, penet_loss_batched_out = torch.zeros(bs).to(device), torch.zeros(bs).to(device)
        thresh = 0.005
        # (*) Object to subject.
        if self.config["obstacle_obj2sbj"]:
            face_vertices = kaolin.ops.mesh.index_vertices_by_faces(sbj_verts, sbj_faces)                                    # (bs, F_sbj, 3, 3)
            dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(obj_verts.contiguous(), face_vertices)           # (bs, N_obj)
            sign = dist < 0.001
            ones = torch.ones_like(sign.long()) 
            sign = torch.where(sign, -ones, ones)
            obj2sbj = dist * sign                                                                                            # (bs, N_obj)
            zeros_o2s, ones_o2s = torch.zeros_like(obj2sbj).to(device), torch.ones_like(obj2sbj).to(device)
            loss_o2s_in = torch.sum(abs(torch.where(obj2sbj<thresh, obj2sbj-thresh, zeros_o2s)), 1) / num_obj_verts          # (bs,) -- averaged across (bs, N_obj)
            loss_o2s_out = torch.sum(torch.log(torch.where(obj2sbj>thresh, obj2sbj+ones_o2s, ones_o2s)), 1) / num_obj_verts  # (bs,) -- averaged across (bs, N_obj)
            penet_loss_batched_in += loss_o2s_in
            penet_loss_batched_out += loss_o2s_out

        # (*) Subject to object.
        if self.config["obstacle_sbj2obj"]:
            face_vertices = kaolin.ops.mesh.index_vertices_by_faces(obj_verts, obj_faces)                                    # (bs, F_obj, 3, 3)
            indices_good_faces = (face_vertices[0].det().abs() > 0.001)                                                      # (F_obj)
            obj_faces = obj_faces[indices_good_faces]
            face_vertices = face_vertices[0][indices_good_faces][None].repeat(bs, 1, 1, 1)                                   # (bs, F_obj_good, 3, 3)
            sign = kaolin.ops.mesh.check_sign(obj_verts, obj_faces, sbj_verts)                                               # (bs, N_sbj)
            dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(sbj_verts.contiguous(), face_vertices)           # (bs, N_sbj)
            ones = torch.ones_like(sign.long()) 
            sign = torch.where(sign, -ones, ones)
            sbj2obj = dist * sign                                                                                            # (bs, N_sbj)
            zeros_s2o, ones_s2o = torch.zeros_like(sbj2obj).to(device), torch.ones_like(sbj2obj).to(device)
            loss_s2o_in = torch.sum(abs(torch.where(sbj2obj<thresh, sbj2obj-thresh, zeros_s2o)), 1) / num_sbj_verts          # (bs,)  -- averaged across (bs, N_sbj)

            if full_body and self.config["obstacle_sbj2obj_extra"] == 'connected_components' and loss_s2o_in.mean() > 0:
                # Connected components based loss.
                edges = np.stack(np.where(adjacency_matrix))
                num_nodes = adjacency_matrix.shape[0]
                v_to_edges = torch.zeros((num_nodes, edges.shape[1]))
                v_to_edges[edges[0], range(edges.shape[1])] = 1
                v_to_edges[edges[1], range(edges.shape[1])] = 1
                indices_inter = (sbj2obj < thresh)
                v_to_edges = v_to_edges[None].expand(bs, -1, -1).clone()
                v_to_edges[torch.where(indices_inter)] = 0
                edges_indices = v_to_edges.sum(1) == 2
                num_inter_v = indices_inter.sum(-1)

                for i in range(bs):
                    if loss_s2o_in[i] > 0:
                        edges_i = edges[:, edges_indices[i]]
                        adj = to_scipy_sparse_matrix(edges_i, num_nodes=num_nodes)
                        n_components, labels = sp.csgraph.connected_components(adj)

                        n_components -= num_inter_v[i]  # Inside obstacles are not taken into account
                        if n_components > 1:
                            indices_out = torch.ones([num_sbj_verts])
                            indices_out[indices_inter[i]] = 0
                            labels_ = labels[indices_out.bool()]
                            # We penalize only the vertices that are out, but the penalization is wrt the original
                            # edge, not including the threshold.
                            most_common_label = Counter(labels_).most_common()[0][0]
                            penalized_joints = (labels != most_common_label) * indices_out.bool().numpy()
                            loss_s2o_in[i] += sbj2obj[i][penalized_joints].sum() / num_sbj_verts

            penet_loss_batched_in += loss_s2o_in
        return penet_loss_batched_in


    def get_human_penet_loss(self, body_verts, body_faces, obstacle_list):
        # Preliminaries.
        batch = body_verts.shape[0]
        body_faces = self.sbj_faces_simplified                                                      
        body_verts = body_verts[:, self.sbj_verts_id, :]                                             
        adjacency_matrix = self.adj_matrix_simplified

        # Compute loss for each obstacle.
        obstacle_loss_batched_in, obstacle_loss_batched_out = torch.zeros(batch).to(self.device), torch.zeros(batch).to(self.device)
        for obstacle in obstacle_list:
            olb_in = self.intersection(
                body_verts, 
                torch.tensor(obstacle['vertices'][None]).to(torch.float32).to(self.device), 
                body_faces, 
                torch.tensor(obstacle['faces']).to(torch.int64).to(self.device), 
                True, 
                adjacency_matrix
            )
            obstacle_loss_batched_in += olb_in
        if len(obstacle_list):
            obstacle_loss_batched_in /= len(obstacle_list)
        
        return obstacle_loss_batched_in
    
    def get_floor_align_loss(self, body_verts):
        values,_ = body_verts[:,:,2].min(1)
        return torch.abs(values)

    def get_smplx_hand_mano_hand_loss(self,body_verts, hand_verts, factor=10):
        smplx_hand_verts = body_verts[:, self.correspondences]
        loss = ((smplx_hand_verts - hand_verts)**2).mean(2).mean(1)
        mask = self.interesting['interesting_vertices_larger']
        body_verts_interest = smplx_hand_verts[:, mask]
        hand_verts_interest = hand_verts[:, mask]
        loss_interest = ((body_verts_interest - hand_verts_interest)**2).mean(2).mean(1)
        loss += factor * loss_interest    
        return loss
 
    
    def get_body_pose_loss(self, predicted_bpose, initial_bpose):
        return ((predicted_bpose - initial_bpose)**2).mean(1)
    

    def get_gaze_loss(self, body_verts, obj_transl):
        batch = body_verts.shape[0]
        head_front, head_back = body_verts[:, 8970], body_verts[:, 8973] 
        obj_transl = obj_transl.repeat(batch, 1) 

        vec_head, vec_obj = head_front - head_back, obj_transl - head_back
        norm_head = torch.norm(vec_head, dim=1) + 1e-4   
        norm_obj = torch.norm(vec_obj, dim=1)+ 1e-4     
        vec_head = vec_head / norm_head[...,None]
        vec_obj = vec_obj / norm_obj[...,None]

        dot_prod = torch.bmm(vec_head.view(batch, 1, -1), vec_obj.view(batch, -1, 1))[:, 0, 0]
        if batch == 1 and dot_prod > 1:
            dot_prod = torch.tensor(1).to(body_verts.device)
        gaze_loss = torch.arccos(dot_prod)
        return gaze_loss
   

    def get_ground_loss(self, vertices):
        bs = vertices.shape[0]
        vertices_hd = self.hdfy_op.hdfy_mesh(vertices)
        # get vertices under the ground plane
        ground_plane_height = 0.0  # obtained by visualization on the presented pose
        vertex_height = (vertices_hd[:, :, 2] - ground_plane_height)
        inside_mask = vertex_height < 0.00
        outside_mask = vertex_height >= 0.00
        v2v_push = (self.b1 * torch.tanh((vertex_height * inside_mask) / self.b2)**2)
        return v2v_push
    
    def get_posture_loss(self, init_vec, pred_vec):
        return init_vec @ pred_vec.T


    def total_loss(
        self, 
        body_verts, 
        hand_verts,
        body_faces, 
        obstacle_list,
        pred_bpose,
        init_bpose,
        init_human_vec,
        pred_human_vec,
        obj_transl,
        factor):
      
        penet_loss_in = self.get_human_penet_loss(body_verts, body_faces, obstacle_list)

        floor_loss = self.get_floor_align_loss(body_verts)
        
        bpose_loss = self.get_body_pose_loss(pred_bpose, init_bpose)

        hand_loss = self.get_smplx_hand_mano_hand_loss(body_verts, hand_verts, factor) 

        gaze_loss = self.get_gaze_loss(body_verts, obj_transl)
        
        under_ground_loss = self.get_ground_loss(body_verts)

        posture_loss = - self.get_posture_loss(init_human_vec, pred_human_vec)
        
        
        losses_dict = {"penetration_loss_in": penet_loss_in,
                       "floor_loss": floor_loss,
                       "bpose_loss": bpose_loss,
                       "hand_loss": hand_loss,
                       "gaze_loss": gaze_loss,               
                       "posture_loss": posture_loss,
                       "under_ground_loss": under_ground_loss.mean(1)
        }
        
        return losses_dict


    def final_loss(
        self,
        body_verts,
        hand_verts,
        body_faces,
        factor,
        pred_bpose,
        init_bpose,
        obj_transl,
        int_sh_pose_pred,
        int_sh_pose_old,
        int_el_pose_pred,
        int_el_pose_old,
        int_wr_pose_pred,
        int_wr_pose_old):
    
        hand_loss = self.get_smplx_hand_mano_hand_loss(body_verts, hand_verts,factor) 
        spose_loss = self.get_body_pose_loss(int_sh_pose_pred, int_sh_pose_old)
        elpose_loss = self.get_body_pose_loss(int_el_pose_pred, int_el_pose_old)
        wpose_loss = self.get_body_pose_loss(int_wr_pose_pred, int_wr_pose_old)
       
        losses_dict = {"hand_loss": hand_loss,
                       "sh_loss": spose_loss,
                       "el_loss": elpose_loss,
                       "wr_loss": wpose_loss,
                        
        }    

        return losses_dict


        

    
