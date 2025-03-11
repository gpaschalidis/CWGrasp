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
import sys
import open3d as o3d
import torch
import argparse
import os
import mano
import time
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from tqdm import tqdm
import kaolin
import pickle as pkl
from vis_utils import rotmat2aa, aa2rotmat, read_o3d_mesh, create_o3d_mesh, \
            create_o3d_box_mesh, create_line_set

from reachingfield.reachingfield import ReachingField
from cgrasp.tools.cfg_parser import Config
from cgrasp.test.tester import Tester
from cgrasp.test.grasp import Grasp
from creach.models.cvae import CReach
from creach.test.reaching_body import ReachingBody
from optimize.human import Human


def main(argv):
    parser = argparse.ArgumentParser(
        description="Optimize usign Reaching Field sampled rays"
    )   

    parser.add_argument(
        "--obj_rec_conf",
        required=True,
        help="The name of the receptacle configuration that is going to be used."
    )   
    parser.add_argument(
        "--config_file",
        required=True,
        help="The path of the configuration file for the optimization script"
    )   
    parser.add_argument(
        "--gender",
        default="male",
        choices=["male","female","neutral"],
        help="Specify the gender"
    )   
    parser.add_argument(
        "--grasp_type",
        default="right",
        choices=["right", "left"],
        help="The path to the folder containing MANO RIGHT model"
    )   
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="The path to the folder containing MANO RIGHT model"
    )   
    parser.add_argument(
        "--save_path",
        required=True,
        help="Specify the file where the results will be stored"
    ) 

    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Working on:{}".format(device))
    

    obj_rec_conf = args.obj_rec_conf

    with open(args.config_file, "r") as f:
        config_optimization = yaml.load(f, Loader=Loader)

    weights = config_optimization["weights"]
    config = config_optimization["data"] 
    
    rhm_path = config["mano_model_path"]
    smplx_path = config["smplx_path"]
    grasp_type = args.grasp_type
    num_samples = args.num_samples
    gender = args.gender


    obj_name = obj_rec_conf.split("_")[0]
    rec_name = "_".join(obj_rec_conf.split("_")[1:-2])
    obj_path = os.path.join(os.getcwd(),"data/contact_meshes", "{}.ply".format(obj_name))
    metadata_path = os.path.join(os.getcwd(),"data/replicagrasp", "dset_info.npz")
    metadata = dict(np.load(metadata_path, allow_pickle=True))[obj_rec_conf]
    obj_transl = metadata[0]
    global_orient = metadata[1]
    obj_mesh = read_o3d_mesh(obj_path)
    obj_verts = np.array(obj_mesh.vertices) @ global_orient.T + obj_transl
    obj_center = (obj_verts.max(0) + obj_verts.min(0)) / 2
    obj_faces = np.array(obj_mesh.triangles)
    obj_mesh = create_o3d_mesh(obj_verts, obj_faces, (0.3, 0.8, 0.1))

    rec_data_path = os.path.join(os.getcwd(),"data/replicagrasp", "receptacles.npz")
    rec_data = dict(np.load(rec_data_path, allow_pickle=True))[rec_name]
    rec_verts = rec_data[0][0]
    rec_faces = rec_data[0][1]
    rec_list = [{"vertices":rec_verts, "faces": rec_faces}] 
    rec_mesh = create_o3d_mesh(rec_verts, rec_faces, (0.3,0.1,0.5))
    floor_mesh = create_o3d_box_mesh(rec_verts)        

    reachingfield = ReachingField(obj_path)
    ray_dirs, hor_ray_dirs = reachingfield.sample(
                obj_transl, 
                global_orient, 
                [rec_mesh], 
                grasp_type=grasp_type,
                num_samples=num_samples
    )
        

    #option = args.option
    
    ################ For the CGrasp ############################
    config_pretrained = config_optimization["pretrained"]
    save_dir = args.save_path
    cgrasp_model_path = config_pretrained["cgrasp"]
    best_rnet = config_pretrained["refine_net"]
    bps_dir = config["bps_dir"]
    closed_mano_faces_path = config["mano_closed_faces_path"] 

    if grasp_type == "right":
        closed_mano_faces = dict(np.load(closed_mano_faces_path))["rh_faces"]
    elif grasp_type == "left":
        closed_mano_faces = dict(np.load(closed_mano_faces_path))["lh_faces"]

    cgrasp_config = { 
        'bps_dir': bps_dir,
        'rhm_path': rhm_path,
        'save_dir':save_dir,
        "cgrasp_model_path": cgrasp_model_path,
        'best_rnet': best_rnet,
        'latentD': config["latentD"]
    }   
    
    
    with open(config["mano2smplx_verts_ids_both"],"rb") as f :
        correspondences = pkl.load(f)

    hand_corr = correspondences["{}_hand".format(grasp_type)]

    cfg = Config(**cgrasp_config)
    cgrasp_tester = Tester(cfg=cfg)
    grasp = Grasp(
        cgrasp_tester, obj_path, save_dir, closed_mano_faces_path, scale=1
    )
    grasp.set_input_params(n_samples=num_samples,   
                           rotmat=global_orient,
                           direction=-ray_dirs,
                           grasp_type=grasp_type) 

    drec_rnet, hand_verts = grasp.generate_grasps()
    
    hand_verts += torch.tensor(obj_transl).to(device)
    
    hand_meshes = [create_o3d_mesh(hand_verts[i].cpu().numpy(), closed_mano_faces, (0.5,0.1,0.8)) for i in range(num_samples)]    
    
    for i in range(num_samples):
        line_set = create_line_set(obj_center[None], obj_center[None] + 2 * ray_dirs[i], [1, 0, 1])
        o3d.visualization.draw_geometries([obj_mesh,floor_mesh,rec_mesh,line_set,hand_meshes[i]])


    network = CReach().to(device)
    network.load_state_dict(torch.load("pretrained/creach.pt", map_location=device), strict=False)
    network.eval()

    reaching_body = ReachingBody(
        network, obj_mesh, obj_center, metadata, [rec_mesh, floor_mesh], smplx_path, device
    )
    body_params = reaching_body.generate(
        num_samples, grasp_type, ray_dirs, hor_ray_dirs, gender, vis=True
    )

    
    
    if grasp_type == "right":            
        body_params["right_hand_pose"] = drec_rnet["hand_pose"].unsqueeze(0).to(device)
    elif grasp_type == "left":
        hand_pose_rotmat = aa2rotmat(drec_rnet["hand_pose"]).squeeze().reshape(-1,15,3,3)
        M = np.eye(3)
        M[0][0] = -1
        left_pose_rotmat = M @ hand_pose_rotmat.cpu().numpy() @ M 
        left_hand_pose = rotmat2aa(torch.tensor(left_pose_rotmat)).reshape(-1,45)
        body_params["left_hand_pose"] = left_hand_pose.to(torch.float32).to(device)


    human = Human(
        body_params, 
        gender,
        grasp_type, 
        smplx_path, 
        num_samples,
        hand_verts,
        rec_list,
        obj_transl,
        config,
        device
    )        


    for param in human.sbj_m.parameters():
        param.requires_grad = False

    bm  = human.sbj_m(**human.body_params)
    body_verts = bm.vertices.detach().cpu().numpy()
    body_joints = bm.joints.detach().cpu().numpy()
    old_verts = body_verts.copy()
    
    minimum_height = old_verts[0].min(0)[2]

    mean_foot = (body_joints[:,8] + body_joints[:,7])/2
    pelvis = body_joints[:,0]
    human_vec = pelvis - mean_foot
    human_vec = human_vec / (np.sqrt((human_vec**2).sum(1))[...,None] +  1e-10)

    human.human_vec = torch.tensor(human_vec).to(torch.float32).to(device)

    hor_ray_dirs = torch.tensor(hor_ray_dirs).to(device)

    obst_verts = torch.tensor(rec_verts).to(torch.float32).to(device)[None]
    obst_faces = torch.tensor(rec_faces).to(torch.int64).to(device)
    sbj_verts = torch.tensor(old_verts).to(torch.float32).to(device)
    sign = kaolin.ops.mesh.check_sign(obst_verts, obst_faces, sbj_verts)
    inter_num = (sign == True).sum()
    if inter_num > old_verts.shape[1] // 10 and obj_center[2] < 0.25:
        human.body_params["transl"] += hor_ray_dirs
        tbm = human.sbj_m(**human.body_params)
        old_verts = tbm.vertices.detach().cpu().numpy()
        translated_bodies = [create_o3d_mesh(old_verts[hu], human.faces.cpu(), list(np.random.rand(1,3)[0])) for hu in range(num_samples)]       
    human.set_opt_parameters()
     
    optimizer = torch.optim.Adam(human.parameters(), lr=0.01)

    weight_penet_in = weights["penet_in"]
    weight_floor = weights["floor"]
    weight_body_pose = weights["body_pose"]
    weight_hand = weights["hand"]
    weight_gaze = weights["gaze"]
    weight_posture = weights["posture"] 
    weight_under_ground = weights["under_ground"]      
    if minimum_height >  0:
        weight_under_ground = 0
    total_loss = [] 

    step = 800
    start_time = time.time()
    for i in tqdm(range(1500)):
        if i < step:
            total_loss_dict = human()
         
            penet_loss_in = weight_penet_in * total_loss_dict["penetration_loss_in"]
            floor_loss = weight_floor * total_loss_dict["floor_loss"]
            bpose_loss = weight_body_pose * total_loss_dict["bpose_loss"]
            hand_loss = weight_hand * total_loss_dict["hand_loss"]
            gaze_loss = weight_gaze * total_loss_dict["gaze_loss"]
            posture_loss = weight_posture * total_loss_dict["posture_loss"]
            under_ground_loss = weight_under_ground * total_loss_dict["under_ground_loss"]
            loss = bpose_loss + hand_loss + gaze_loss + penet_loss_in + posture_loss + under_ground_loss + floor_loss 
        else:
            if i == step:
                human.reset_opt_parameters()
                optimizer.add_param_group({
                    "params":[
                        human.int_wrist_pose,
                        human.int_elbow_pose,
                        human.int_shoulder_pose
                    ]
                })
                weight_hand = 100 * weights["hand"]
        
            total_loss_dict = human.final_step()
        
            hand_loss = weight_hand * total_loss_dict["hand_loss"]
            shpose_loss = total_loss_dict["sh_loss"]
            elpose_loss = 0.8 * total_loss_dict["el_loss"]
            wrpose_loss = 0.6 * total_loss_dict["wr_loss"]
            loss = hand_loss + shpose_loss + elpose_loss + wrpose_loss
        mean_loss = loss.mean()
        
        optimizer.zero_grad()
        mean_loss.backward()
        optimizer.step()

    new_bm  = human.sbj_m(**human.body_params)
    body_verts = new_bm.vertices.detach().cpu().numpy()
    body_faces = human.faces.cpu()
    hand_verts = body_verts[:, hand_corr]
    opt_hand_mesh = [create_o3d_mesh(hand_verts[j], closed_mano_faces, (0.5,0.9,0.1)) for j in range(num_samples)]  

    sbj_mesh_new = [create_o3d_mesh(body_verts[j], human.faces.cpu(), (0.1,0.7,0.5)) for j in range(num_samples)]
    
    for u in range(num_samples):
        o3d.visualization.draw_geometries([obj_mesh, floor_mesh, rec_mesh, sbj_mesh_new[u]])
        o3d.visualization.draw_geometries([obj_mesh, opt_hand_mesh[u], hand_meshes[u]])
    
    np.savez(os.path.join(save_dir,"results.npz"),
        human_verts=body_verts,
        human_faces=body_faces,
        hand_verts=hand_verts,
        hand_faces=closed_mano_faces)

if __name__ == "__main__":
    main(sys.argv[1:])
                            
