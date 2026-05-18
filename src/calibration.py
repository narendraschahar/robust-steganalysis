import torch
import torch.nn as nn

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))
    
    @property
    def temperature(self):
        return torch.exp(self.log_temperature)
    
    def forward(self, logits):
        return logits / self.temperature

def fit_temperature(model, val_loader, device):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(device)
            logits_list.append(model(x).detach())
            labels_list.append(y.to(device))
            
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    scaler = TemperatureScaler().to(device)
    
    opt = torch.optim.LBFGS([scaler.log_temperature], lr=0.05, max_iter=200)
    ce = nn.CrossEntropyLoss()
    
    def closure():
        opt.zero_grad()
        loss = ce(scaler(logits), labels)
        loss.backward()
        return loss
        
    opt.step(closure)
    return float(scaler.temperature.detach().cpu().item())
