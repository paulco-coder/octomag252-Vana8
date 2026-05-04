import numpy as np
import random
import math

def generate_data_v1(n_samples, sequence_length):
    batch_x, batch_y = [], []
    for _ in range(n_samples):
        rand = random.random() * 2 * math.pi
        sig1 = np.sin(np.linspace(0.0 * math.pi + rand, 3.0 * math.pi + rand, sequence_length * 2))
        sig2 = np.cos(np.linspace(0.0 * math.pi + rand, 3.0 * math.pi + rand, sequence_length * 2))
        
        x_ = np.array([sig1[:sequence_length], sig2[:sequence_length]]).T
        y_ = np.array([sig1[sequence_length:], sig2[sequence_length:]]).T
        
        batch_x.append(x_)
        batch_y.append(y_)
    return np.array(batch_x), np.array(batch_y)

def generate_data_v2(n_samples, seq_length):
    batch_x, batch_y = [], []
    for _ in range(n_samples):
        offset_1 = random.random() * 2 * math.pi
        freq_1 = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_1 = random.random() + 0.1
        sig1 = amp_1 * np.sin(np.linspace(
            seq_length / 15.0 * freq_1 * 0.0 * math.pi + offset_1,
            seq_length / 15.0 * freq_1 * 3.0 * math.pi + offset_1, seq_length * 2))

        offset_2 = random.random() * 2 * math.pi
        freq_2 = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_2 = random.random() * 1.2
        sig2 = amp_2 * np.cos(np.linspace(
            seq_length / 15.0 * freq_2 * 0.0 * math.pi + offset_2,
            seq_length / 15.0 * freq_2 * 3.0 * math.pi + offset_2, seq_length * 2)) + sig1

        batch_x.append(np.array([sig2[:seq_length]]).T)
        batch_y.append(np.array([sig2[seq_length:]]).T)
    return np.array(batch_x), np.array(batch_y)

def generate_data_v3(n_samples, seq_length):
    x, y = generate_data_v2(n_samples, seq_length)
    noise_amount = random.random() * 0.15 + 0.10
    x = x + noise_amount * np.random.randn(n_samples, seq_length, 1)

    avg = np.average(x)
    std = np.std(x) + 0.0001
    x = (x - avg) / std / 2.5
    y = (y - avg) / std / 2.5

    return x, y

def generate_data_v4(n_samples, seq_length, hole_start, hole_length):
    batch_x, batch_y = [], []
    for _ in range(n_samples):
        offset_1 = random.random() * 2 * math.pi
        freq_1 = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_1 = random.random() + 0.1
        sig1 = amp_1 * np.sin(np.linspace(
            seq_length / 15.0 * freq_1 * 0.0 * math.pi + offset_1,
            seq_length / 15.0 * freq_1 * 3.0 * math.pi + offset_1, seq_length))

        offset_2 = random.random() * 2 * math.pi
        freq_2 = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_2 = random.random() * 1.2
        sig2 = amp_2 * np.cos(np.linspace(
            seq_length / 15.0 * freq_2 * 0.0 * math.pi + offset_2,
            seq_length / 15.0 * freq_2 * 3.0 * math.pi + offset_2, seq_length)) + sig1

        part1 = sig2[:hole_start]
        part2 = sig2[hole_start:hole_start+hole_length]
        part3 = sig2[hole_start+hole_length:]
        
        x_ = np.concatenate([part1, part3])
        y_ = part2
        
        batch_x.append(np.expand_dims(x_, axis=-1))
        batch_y.append(np.expand_dims(y_, axis=-1))
        
    return np.array(batch_x), np.array(batch_y)

def generate_data_v5(n_samples):
    max_ctx = 300
    max_hole = 50
    total_max_len = max_ctx * 2 + max_hole
    
    batch_x = np.zeros((n_samples, max_ctx * 2, 1))
    batch_y = np.zeros((n_samples, max_hole, 1))
    batch_mask = np.zeros((n_samples, max_hole, 1))
    
    for i in range(n_samples):
        ctx_l = random.randint(100, max_ctx)
        ctx_r = random.randint(100, max_ctx)
        hole = random.randint(5, max_hole)
        
        offset_1 = random.random() * 2 * math.pi
        freq_1 = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_1 = random.random() + 0.1
        sig1 = amp_1 * np.sin(np.linspace(
            total_max_len / 15.0 * freq_1 * 0.0 * math.pi + offset_1,
            total_max_len / 15.0 * freq_1 * 3.0 * math.pi + offset_1, total_max_len))

        offset_2 = random.random() * 2 * math.pi
        freq_2 = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_2 = random.random() * 1.2
        t2 = np.linspace(
            total_max_len / 15.0 * freq_2 * 0.0 * math.pi + offset_2,
            total_max_len / 15.0 * freq_2 * 3.0 * math.pi + offset_2, total_max_len)
            
        trend_slope = (random.random() - 0.5) * 2.0
        trend = np.linspace(0, trend_slope, total_max_len)
        noise = np.random.randn(total_max_len) * 0.15
        
        sig_final = sig1 + amp_2 * (np.cos(t2) ** 3) + trend + noise
        
        start_hole = max_ctx
        
        part1 = sig_final[start_hole - ctx_l : start_hole]
        part2 = sig_final[start_hole : start_hole + hole]
        part3 = sig_final[start_hole + hole : start_hole + hole + ctx_r]
        
        batch_x[i, max_ctx - ctx_l : max_ctx, 0] = part1
        batch_x[i, max_ctx : max_ctx + ctx_r, 0] = part3
        batch_y[i, :hole, 0] = part2
        batch_mask[i, :hole, 0] = 1.0
        
    return batch_x, batch_y, batch_mask

def generate_real_sensor_signals(n_signals=3, base_duration_sec=0.2):
    fs = 100000 
    signals = []
    
    for _ in range(n_signals):
        dur = base_duration_sec * random.uniform(0.8, 1.2)
        n_pts = int(dur * fs)
        t = np.linspace(0, dur, n_pts)
        
        f_start = random.uniform(1000, 2000)
        f_end = random.uniform(2000, 3000)
        
        freqs = np.linspace(f_start, f_end, n_pts)
        phase = np.cumsum(freqs) / fs * 2 * math.pi
        
        sig = 0.1 * np.sin(phase) + 0.2 * (np.cos(2*phase)**3) + 0.1 * np.sin(0.5*phase)
        trend = np.linspace(0, random.uniform(-1, 1), n_pts)
        noise = np.random.randn(n_pts) * 0.1
        sig = sig + trend + noise
        
        hole_ratio = random.uniform(0.05, 0.25)
        is_hole = (phase % (2 * math.pi)) < (2 * math.pi * hole_ratio)
        
        sig_corrompu = sig.copy()
        sig_corrompu[is_hole] = np.nan
        signals.append((t, sig_corrompu, sig))
        
    return signals

def build_self_supervised_dataset(signals, max_ctx=250, max_hole=70):
    X, Y, M = [], [], []
    for t_arr, sig_nan, _ in signals:
        is_valid = ~np.isnan(sig_nan)
        padded = np.concatenate(([0], is_valid.view(np.int8), [0]))
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        
        for st, en in zip(starts, ends):
            segment_len = en - st
            min_req = 15
            if segment_len > min_req:
                n_samples_from_segment = segment_len // 5
                
                for _ in range(max(1, n_samples_from_segment)):
                    sim_hole_len = random.randint(3, min(max_hole, segment_len // 3))
                    sim_hole_st = random.randint(st + 3, en - sim_hole_len - 3)
                    
                    ctx_l_len = min(max_ctx, sim_hole_st - st)
                    ctx_r_len = min(max_ctx, en - (sim_hole_st + sim_hole_len))
                    
                    part1 = sig_nan[sim_hole_st - ctx_l_len : sim_hole_st]
                    part2 = sig_nan[sim_hole_st : sim_hole_st + sim_hole_len]
                    part3 = sig_nan[sim_hole_st + sim_hole_len : sim_hole_st + sim_hole_len + ctx_r_len]
                    
                    x_ = np.zeros((max_ctx * 2, 1))
                    y_ = np.zeros((max_hole, 1))
                    m_ = np.zeros((max_hole, 1))
                    
                    x_[max_ctx - ctx_l_len : max_ctx, 0] = part1
                    x_[max_ctx : max_ctx + ctx_r_len, 0] = part3
                    y_[:sim_hole_len, 0] = part2
                    m_[:sim_hole_len, 0] = 1.0
                    
                    X.append(x_)
                    Y.append(y_)
                    M.append(m_)
                    
    return np.array(X), np.array(Y), np.array(M)