import torch
import torch.optim as optim
import numpy as np
from config import device

def train_model(model, train_x, train_y, val_x, val_y, epochs=15, batch_size=100, lr=0.001, wd=0.0003):
    def criterion(outputs, targets):
        return torch.mean(torch.sum((outputs - targets) ** 2, dim=[1, 2]) / 2.0)
        
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n_samples = train_x.shape[0]
    
    train_history = []
    val_history = []
    
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_samples)
        total_loss = 0
        num_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = torch.FloatTensor(train_x[batch_idx]).to(device)
            batch_y = torch.FloatTensor(train_y[batch_idx]).to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_train_loss = total_loss / num_batches
        train_history.append(avg_train_loss)
        
        model.eval()
        with torch.no_grad():
            val_x_tensor = torch.FloatTensor(val_x).to(device)
            val_y_tensor = torch.FloatTensor(val_y).to(device)
            val_outputs = model(val_x_tensor)
            val_loss = criterion(val_outputs, val_y_tensor).item()
            val_history.append(val_loss)
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
    return train_history, val_history

def train_model_v5(model, train_x, train_y, train_m, val_x, val_y, val_m, epochs=20, batch_size=128, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n_samples = train_x.shape[0]
    
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_samples)
        total_loss = 0
        num_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = torch.FloatTensor(train_x[batch_idx]).to(device)
            batch_y = torch.FloatTensor(train_y[batch_idx]).to(device)
            batch_mask = torch.FloatTensor(train_m[batch_idx]).to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            loss = torch.mean(torch.sum(((outputs - batch_y) * batch_mask) ** 2, dim=[1, 2]) / 2.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            
        model.eval()
        val_loss_total = 0
        val_batches = 0
        with torch.no_grad():
            for j in range(0, val_x.shape[0], batch_size):
                v_x = torch.FloatTensor(val_x[j:j+batch_size]).to(device)
                v_y = torch.FloatTensor(val_y[j:j+batch_size]).to(device)
                v_m = torch.FloatTensor(val_m[j:j+batch_size]).to(device)
                v_out = model(v_x)
                loss = torch.mean(torch.sum(((v_out - v_y) * v_m) ** 2, dim=[1, 2]) / 2.0).item()
                val_loss_total += loss
                val_batches += 1
        v_loss = val_loss_total / val_batches
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/num_batches:.6f} | Val Loss: {v_loss:.6f}")

def train_model_v6_fft(model, train_x, train_y, train_m, val_x, val_y, val_m, epochs=20, batch_size=128, lr=0.001, alpha=0.1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n_samples = train_x.shape[0]
    
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_samples)
        total_loss = 0
        total_time_loss = 0
        total_freq_loss = 0
        num_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = torch.FloatTensor(train_x[batch_idx]).to(device)
            batch_y = torch.FloatTensor(train_y[batch_idx]).to(device)
            batch_mask = torch.FloatTensor(train_m[batch_idx]).to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            loss_time = torch.mean(torch.sum(((outputs - batch_y) * batch_mask) ** 2, dim=[1, 2]) / 2.0)
            
            fft_out = torch.fft.rfft(outputs * batch_mask, dim=1)
            fft_y = torch.fft.rfft(batch_y * batch_mask, dim=1)
            loss_freq = torch.mean(torch.sum((torch.abs(fft_out) - torch.abs(fft_y)) ** 2, dim=[1, 2]) / 2.0)
            
            loss = loss_time + alpha * loss_freq
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_time_loss += loss_time.item()
            total_freq_loss += loss_freq.item()
            num_batches += 1
            
        model.eval()
        v_loss_total = 0
        val_batches = 0
        with torch.no_grad():
            for j in range(0, val_x.shape[0], batch_size):
                v_x = torch.FloatTensor(val_x[j:j+batch_size]).to(device)
                v_y = torch.FloatTensor(val_y[j:j+batch_size]).to(device)
                v_m = torch.FloatTensor(val_m[j:j+batch_size]).to(device)
                v_out = model(v_x)
                
                v_loss_time = torch.mean(torch.sum(((v_out - v_y) * v_m) ** 2, dim=[1, 2]) / 2.0)
                
                v_fft_out = torch.fft.rfft(v_out * v_m, dim=1)
                v_fft_y = torch.fft.rfft(v_y * v_m, dim=1)
                v_loss_freq = torch.mean(torch.sum((torch.abs(v_fft_out) - torch.abs(v_fft_y)) ** 2, dim=[1, 2]) / 2.0)
                
                v_loss = (v_loss_time + alpha * v_loss_freq).item()
                v_loss_total += v_loss
                val_batches += 1
                
        print(f"Epoch {epoch+1}/{epochs} | Loss Tot: {total_loss/num_batches:.6f} | Val: {v_loss_total/val_batches:.6f}")
