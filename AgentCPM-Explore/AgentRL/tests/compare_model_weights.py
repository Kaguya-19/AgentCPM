from transformers import AutoModelForCausalLM
import torch

model1 = AutoModelForCausalLM.from_pretrained("/share_data/data1/models/MiniCPM-V-4_5-64",torch_dtype=torch.bfloat16, trust_remote_code=True)
model2 = AutoModelForCausalLM.from_pretrained("/data3/workhome/luyaxi/AgentRL/output/gui_agent_trajectory_new/checkpoint-50",torch_dtype=torch.bfloat16, trust_remote_code=True)

sd1 = model1.state_dict()
sd2 = model2.state_dict()
if set(sd1.keys()) != set(sd2.keys()):
    print("State dict keys do not match!")
    print("In model1 but not in model2:", set(sd1.keys()) - set(sd2.keys()))
    print("In model2 but not in model1:", set(sd2.keys()) - set(sd1.keys()))
    
for k in sd1.keys():
    # calculate the norm of the difference
    diff = sd1[k].cuda() - sd2[k].cuda()
    norm = torch.norm(diff).item()
    sd1[k] = sd1[k].cpu()
    sd2[k] = sd2[k].cpu()
    
    if norm > 1e-3:
        # raise Warning(f"Parameter {k} differs: norm of difference = {norm}")
        print(f"Warning: Parameter {k} differs: norm of difference = {norm}")
    else:
        print(f"Parameter {k} matches: norm of difference = {norm}")