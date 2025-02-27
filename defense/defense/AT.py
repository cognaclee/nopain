import torch  

def remove_outliers_fn(x, model_loss_fn, top_k=10, num_std=1.0):    
    dists = x.unsqueeze(1) - x.unsqueeze(2)  
    dists = torch.norm(dists, dim=3)  

    diag = torch.eye(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1, -1)  
    dists = torch.where(diag > 0.0, torch.tensor(float("inf"), device=x.device), dists)  

    _, top_k_dists = torch.topk(-dists, k=top_k, dim=2)  

    mean_dists = top_k_dists.mean(dim=2)  
    avg = mean_dists.mean(dim=1, keepdim=True)  
    var = mean_dists.var(dim=1, keepdim=True)  
    std = num_std * torch.sqrt(var)  
 
    remove = mean_dists > (avg + std)  
    
    idx = remove.float().argmin(dim=1)  
    one_hot = torch.nn.functional.one_hot(idx, num_classes=x.shape[1]).float()  
    replace = (x * one_hot.unsqueeze(2)).sum(dim=1, keepdim=True)  
    
    x = torch.where(remove.unsqueeze(2), replace + torch.zeros_like(x), x)  

    return x.detach() 

def remove_salient_points_fn(x, model_loss_fn, top_k=100):  
    logits, _ = model_loss_fn(x)  
    grads = []  
    for i in range(logits.shape[1]):  
        grad = torch.autograd.grad(logits[:, i].sum(), x, retain_graph=True)[0]  
        grads.append(grad)  
    grads = torch.stack(grads, dim=0)  

    norms = torch.norm(grads, dim=3)  
    max_norms = norms.max(dim=0)[0]  
    _, remove = torch.topk(max_norms, k=top_k, dim=0)  

    remove = torch.zeros(x.size(1), dtype=torch.bool, device=x.device)  
    remove[remove] = True  
    remove_any = remove.any(dim=1)  

    idx = remove_any.float().argmin(dim=1)  
    one_hot = torch.nn.functional.one_hot(idx, num_classes=x.shape[1]).float()  
    replace = (x * one_hot.unsqueeze(2)).sum(dim=1, keepdim=True)  

    x = torch.where(remove_any.unsqueeze(2), replace + torch.zeros_like(x), x)  

    return x.detach() 