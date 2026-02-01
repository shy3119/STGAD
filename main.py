import pickle
import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
from src.analysis import *
from src.plot_abnormal_samples import *
from src.analysis import explain_true_positive
import time as _time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _move_optimizer_state_to_device(optim, device):
    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def normalize_score(score):
    return (score - np.min(score)) / (np.max(score) - np.min(score) + 1e-8)

def compute_anomaly_score(x_real, x_gen, d_score):
    recon_error = torch.mean((x_real - x_gen) ** 2, dim=1)  # [B]
    d_score = d_score.view(-1)  # [B]
    return recon_error.detach().cpu().numpy(), d_score.detach().cpu().numpy()

def find_abnormal_sample(labels):
    for i in range(len(labels)):
        if labels[i].sum() > 0:
            return i
    return None

def add_noise_to_data(data, noise_level):
    noise = torch.randn_like(data) * noise_level
    return data + noise

def convert_to_windows(data, window_size, step_size=1):
    B, T, F = data.shape
    windows = []
    for i in range(0, T - window_size + 1, step_size):
        window = data[:, i:i + window_size, :]  # [B, W, F]
        window = window.reshape(B, -1)  # [B, W*F]
        windows.append(window)
    return torch.cat(windows, dim=0)  # [B*num_windows, W*F]

def load_dataset(dataset):
    folder = os.path.join(output_folder, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    loader = []
    for file in ['train', 'test', 'labels']:
        if dataset == 'SMD': file = 'machine-1-1_' + file
        if dataset == 'SMAP': file = 'P-1_' + file
        if dataset == 'MSL': file = 'C-1_' + file
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    if args.less: loader[0] = cut_array(0.2, loader[0])
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    labels = loader[2]
    return train_loader, test_loader, labels

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
    folder = f'checkpoints/{args.model}_{args.dataset}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims):
    import src.models
    model_class = getattr(src.models, modelname)
    model = model_class(dims).float()
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
    if os.path.exists(fname) and (not args.retrain or args.test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        _move_optimizer_state_to_device(optimizer, DEVICE)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1; accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list

def _sync_if_cuda(dev):
    try:
        import torch
        if dev is not None and hasattr(dev, "type") and dev.type == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass

def benchmark_test_inference(model, test_loader, device, warmup_batches=5, timed_batches=50, n_samples=1):
    import time as _time
    import torch

    model.eval()
    use_cuda = device is not None and hasattr(device, "type") and device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for b_idx, (d,) in enumerate(test_loader):
            if b_idx >= warmup_batches:
                break
            d = d.float().to(device, non_blocking=True)
            for _ in range(n_samples):
                x_gen = model.generate(d.size(0), d.device)
                _ = model.discriminate(d)
                # tiny residual to match real path (cheap)
                _ = (d - x_gen).pow(2).mean(dim=1)

    _sync_if_cuda(device)

    total_windows = 0
    t0 = _time.perf_counter()
    with torch.no_grad():
        for b_idx, (d,) in enumerate(test_loader):
            if b_idx >= timed_batches:
                break
            d = d.float().to(device, non_blocking=True)
            for _ in range(n_samples):
                x_gen = model.generate(d.size(0), d.device)
                _ = model.discriminate(d)
                _ = (d - x_gen).pow(2).mean(dim=1)
            total_windows += int(d.size(0))
    _sync_if_cuda(device)
    t1 = _time.perf_counter()

    elapsed = max(1e-9, (t1 - t0))
    throughput = total_windows / elapsed
    latency_ms_per_window = (elapsed / max(1, total_windows)) * 1000.0

    peak_alloc_gb = None
    peak_reserved_gb = None
    if use_cuda:
        peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

    return {
        "device": str(device),
        "warmup_batches": int(warmup_batches),
        "timed_batches": int(timed_batches),
        "n_samples": int(n_samples),
        "total_windows": int(total_windows),
        "elapsed_s": float(elapsed),
        "throughput_windows_per_s": float(throughput),
        "latency_ms_per_window": float(latency_ms_per_window),
        "peak_alloc_gb": None if peak_alloc_gb is None else float(peak_alloc_gb),
        "peak_reserved_gb": None if peak_reserved_gb is None else float(peak_reserved_gb),
    }

def save_infer_benchmark_csv(row_dict, out_csv):
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        w.writeheader()
        w.writerow(row_dict)

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    alpha = alpha.expand_as(real_samples)

    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)

    fake = torch.ones(d_interpolates.size()).to(device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty



def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True, n_critic=5, lambda_gp=5.0):
    device = next(model.parameters()).device
    l = nn.MSELoss(reduction='mean' if training else 'none')

    optimizer_D = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    if training:
        mses, gls, dls = [], [], []
        is_wgan = 'STG' in model.name
        criterion = nn.BCELoss()

        for (d,) in data:
            d = d.float()
            # Keep batch dimension (do NOT flatten B into feature dim)
            if d.ndim == 1:
                # [input_dim] -> [1, input_dim]
                d = d.view(1, -1)
            elif d.ndim == 2:
                # [B, input_dim] -> keep as-is
                pass
            elif d.ndim == 3:
                # [B, W, F] -> [B, W*F]
                d = d.reshape(d.size(0), -1)
            else:
                raise ValueError(f"Unexpected d.ndim={d.ndim}, shape={tuple(d.shape)}")
            assert d.shape[1] == model.input_dim, f"Input dim mismatch: got {d.shape[1]}, expected {model.input_dim}"
            d = d.to(device, non_blocking=True)
            batch_size = d.size(0)

            for _ in range(n_critic):
                optimizer_D.zero_grad()
                real_validity = model.discriminate(d)
                fake_data = model.generate(batch_size, device)
                fake_validity = model.discriminate(fake_data.detach())

                if is_wgan:
                    d_loss = -(torch.mean(real_validity) - torch.mean(fake_validity))
                    gp = compute_gradient_penalty(model.discriminate, d, fake_data, device)
                    d_loss += lambda_gp * gp
                else:
                    real_labels = torch.ones_like(real_validity)
                    fake_labels = torch.zeros_like(fake_validity)
                    d_loss = criterion(real_validity, real_labels) + criterion(fake_validity, fake_labels)

                d_loss.backward()
                optimizer_D.step()

            optimizer_G.zero_grad()
            fake_data = model.generate(batch_size, device)
            fake_validity = model.discriminate(fake_data)

            mse_loss = nn.MSELoss()(fake_data, d)
            if is_wgan:
                g_loss = -torch.mean(fake_validity)
            else:
                g_loss = criterion(fake_validity, real_labels)

            total_g_loss = g_loss + mse_loss
            total_g_loss.backward()
            optimizer_G.step()

            mses.append(mse_loss.item())
            gls.append(g_loss.item())
            dls.append(d_loss.item())

        tqdm.write(f'Epoch {epoch},	MSES = {np.mean(mses)}')
        return (sum(mses)/len(mses) + sum(gls)/len(gls) + sum(dls)/len(dls)), optimizer_G.param_groups[0]['lr'], mses, gls, dls


    else:
        outputs = []
        processed_data = []

        for (d,) in data:
            d = d.float()
            # Keep batch dimension (do NOT flatten B into feature dim)
            if d.ndim == 1:
                d = d.view(1, -1)
            elif d.ndim == 2:
                pass
            elif d.ndim == 3:
                d = d.reshape(d.size(0), -1)
            else:
                raise ValueError(f"Unexpected d.ndim={d.ndim}, shape={tuple(d.shape)}")

            assert d.shape[1] == model.input_dim, f"Input dim mismatch: got {d.shape[1]}, expected {model.input_dim}"

            d = d.to(device, non_blocking=True)
            fake_data = model.generate(d.size(0), device)
            outputs.append(fake_data)
            processed_data.append(d)

        outputs = torch.cat(outputs, dim=0)
        data_tensor = torch.cat(processed_data, dim=0)
        loss = l(outputs, data_tensor)

        return loss.detach().cpu().numpy(), outputs.detach().cpu().numpy()

if __name__ == '__main__':
    train_loader, test_loader, labels = load_dataset(args.dataset)
    raw_trainD, raw_testD = next(iter(train_loader)), next(iter(test_loader))
    raw_trainD = torch.tensor(raw_trainD).float().unsqueeze(0) if raw_trainD.ndim == 2 else torch.tensor(raw_trainD).float()
    raw_testD = torch.tensor(raw_testD).float().unsqueeze(0) if raw_testD.ndim == 2 else torch.tensor(raw_testD).float()

    window_size = args.window_size if hasattr(args, 'window_size') else 5
    trainD = convert_to_windows(raw_trainD, window_size=window_size)  # shape [N, input_dim]
    testD  = convert_to_windows(raw_testD, window_size=window_size)

    train_dataset = TensorDataset(trainD)
    test_dataset = TensorDataset(testD)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    trainO, testO = trainD.clone(), testD.clone()

    trainO, testO = trainD, testD

    feature_dim = raw_trainD.shape[-1]
    input_dim = window_size * feature_dim
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, input_dim)

    if args.noise_train:
        sample_idx = 0
        original_sample = trainD[sample_idx].cpu().numpy()
        noisy_sample = add_noise_to_data(trainD[sample_idx:sample_idx+1], noise_level=0.25).squeeze().cpu().numpy()

        plt.figure(figsize=(12, 4))
        plt.plot(original_sample, label="Original")
        plt.plot(noisy_sample, label=f"Noisy (level={0.25})", alpha=0.7)
        plt.title(f"Example of Noise Injection (Sample {sample_idx})")
        plt.legend()
        plt.grid(True)
        plt.show()

    if not args.test:
        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        num_epochs = 5; e = epoch + 1; start = time()
        all_mses, all_gls, all_dls = [], [], []
        noise_level = 0.25
        train_bs = int(getattr(args, 'batch_size', 64))
        pin_mem = bool(torch.cuda.is_available())
        base_train_dataset = TensorDataset(trainD)
        base_train_loader = DataLoader(base_train_dataset, batch_size=train_bs, shuffle=True, pin_memory=pin_mem)

        BENCH_TRAIN = bool(getattr(args, 'benchmark', False))

        train_bench_rows = []
        for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
            cur_loader = base_train_loader
            if args.noise_train:
                noisy_trainD = add_noise_to_data(trainD, noise_level=noise_level)
                noisy_dataset = TensorDataset(noisy_trainD)
                cur_loader = DataLoader(noisy_dataset, batch_size=train_bs, shuffle=True, pin_memory=pin_mem)

            if BENCH_TRAIN:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                _t0 = _time.perf_counter()

            lossT, lr, mses, gls, dls = backprop(e, model, cur_loader, trainO, optimizer, scheduler)

            if BENCH_TRAIN:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _t1 = _time.perf_counter()
                _epoch_time = _t1 - _t0
                _peak = (torch.cuda.max_memory_allocated()/(1024**3)) if torch.cuda.is_available() else None
                train_bench_rows.append({'epoch': int(e), 'epoch_time_s': float(_epoch_time), 'peak_mem_gb': _peak})
                print(f"[train-benchmark] epoch={e} time_s={_epoch_time:.3f} peak_mem_gb={_peak if _peak is not None else 'NA'}")

            accuracy_list.append((lossT, lr))
            all_mses.extend(mses)
            all_gls.extend(gls)
            all_dls.extend(dls)

        print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
        save_model(model, optimizer, scheduler, e, accuracy_list)

        plt.figure(figsize=(12,6))
        plt.plot(all_mses, label='MSE Loss')
        plt.plot(all_gls, label='Generator Loss')
        plt.plot(all_dls, label='Discriminator Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title(f'Loss Curves for {args.model} on {args.dataset}')
        plt.legend()
        plt.grid()
        plt.show()

    torch.zero_grad = True
    model.eval()
    print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')

    # ---- Inference benchmark (optional) ----
    if getattr(args, 'benchmark', False):
        dev = model.device if hasattr(model, 'device') else next(model.parameters()).device
        bench = benchmark_test_inference(model, test_loader, dev, warmup_batches=5, timed_batches=50, n_samples=1)
        print(f"[infer-benchmark] device={bench['device']} n_samples={bench['n_samples']} "
              f"throughput={bench['throughput_windows_per_s']:.1f} win/s "
              f"latency={bench['latency_ms_per_window']:.4f} ms/win "
              f"peak_alloc_gb={bench['peak_alloc_gb']} peak_reserved_gb={bench['peak_reserved_gb']}")
        out_csv = f"benchmark_infer_{args.model}_{args.dataset}.csv"
        save_infer_benchmark_csv(bench, out_csv)
        print(f"[infer-benchmark] saved {out_csv}")

    recon_list = []
    dscore_list = []
    all_preds = []
    all_labels = []

    for (d,) in test_loader:
        d = d.float().to(model.device) if hasattr(model, "device") else d.float().to(next(model.parameters()).device)
        x_gen = model.generate(d.size(0), d.device)
        d_score = model.discriminate(d)

        recon_error = torch.mean((d - x_gen) ** 2, dim=1)  # [B]
        d_score = d_score.view(-1)  # [B]

        recon_list.append(recon_error.detach().cpu().numpy())
        dscore_list.append(d_score.detach().cpu().numpy())
        all_preds.append(x_gen.detach().cpu().numpy())
        all_labels.append(d.detach().cpu().numpy())

    recon_all = np.concatenate(recon_list, axis=0)    # [N]
    dscore_all = np.concatenate(dscore_list, axis=0)  # [N]

    recon_norm = (recon_all - recon_all.min()) / (recon_all.max() - recon_all.min() + 1e-8)
    dscore_norm = (dscore_all - dscore_all.min()) / (dscore_all.max() - dscore_all.min() + 1e-8)
    dscore_norm = -dscore_norm

    beta_list = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    def fuse_scores(recon_norm, dscore_norm, beta_recon: float):
        return beta_recon * recon_norm + (1.0 - beta_recon) * dscore_norm

    results_beta = []

    for beta_recon in beta_list:
        final_score = fuse_scores(recon_norm, dscore_norm, beta_recon)  # [N]
        loss = final_score.reshape(-1, 1)

    loss = final_score.reshape(-1, 1)
    y_pred = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    if hasattr(args, "explain") and args.explain:
        #run_attention_analysis(model, testD, sample_index=0)
        abnormal_index = find_abnormal_sample(labels)
        feature_dim = int(model.input_dim // window_size)
        run_attention_analysis(model, testD, labels, window_size, feature_dim, sample_index=abnormal_index, out_dir=f'explain_{args.model}_{args.dataset}')
        explain_true_positive(
            model=model,
            flat_dataset_tensor=testD,    
            labels_np=labels,
            window_size=window_size,
            feature_dim=feature_dim,
            out_dir=f'explain_tp_{args.model}_{args.dataset}',
            threshold_method="pot",
            topk_feat=5,
            also_draw_attn=True
        )
    recon_list_T = []
    dscore_list_T = []

    for (d,) in train_loader:
        d = d.float().to(model.device) if hasattr(model, "device") else d.float().to(next(model.parameters()).device)
        x_gen = model.generate(d.size(0), d.device)
        d_score = model.discriminate(d)

        recon_error = torch.mean((d - x_gen) ** 2, dim=1)  # [B]
        d_score = d_score.view(-1)  # [B]

        recon_list_T.append(recon_error.detach().cpu().numpy())
        dscore_list_T.append(d_score.detach().cpu().numpy())

    recon_all_T = np.concatenate(recon_list_T, axis=0)
    dscore_all_T = np.concatenate(dscore_list_T, axis=0)

    recon_T_norm = (recon_all_T - recon_all_T.min()) / (recon_all_T.max() - recon_all_T.min() + 1e-8)
    dscore_T_norm = (dscore_all_T - dscore_all_T.min()) / (dscore_all_T.max() - dscore_all_T.min() + 1e-8)
    dscore_T_norm = -dscore_T_norm

    for beta_recon in beta_list:
        lossT = fuse_scores(recon_T_norm, dscore_T_norm, beta_recon)
        lossT = lossT.reshape(-1, 1)

    T = min(loss.shape[0], labels.shape[0])
    if (labels.shape[0] != T) or (loss.shape[0] != T):
        print(f"[Align] Truncate to T={T} for loss/labels")
        loss   = loss[:T, :]
        labels = labels[:T, :]

    df = pd.DataFrame()
    preds = []
    max_dims = min(loss.shape[1], labels.shape[1], lossT.shape[1])
    for i in range(max_dims):
        lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
        if len(l) != len(ls):
            print(f"[Warning] Skipping dim {i}: loss and label length mismatch ({len(l)} vs {len(ls)})")
            continue
        try:
            print("lt:", lt.shape)
            print("l:", l.shape)
            print("ls:", ls.shape)
            result, pred = pot_eval(lt, l, ls)
            preds.append(pred)
            #df = df.append(result, ignore_index=True)
            df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
        except ValueError as ve:
            print(f"[Error] pot_eval failed at dim {i}: {ve}")
    if len(preds) > 0:
        T = labels.shape[0]
        preds = [p[:T] for p in preds]
        pred_labels = np.column_stack(preds)  # shape [T, F]
    else:
        pred_labels = None
    if pred_labels is not None:
        T, F = labels.shape[0], labels.shape[1]
        pred_labels = pred_labels[:T]
        if pred_labels.ndim == 1:
            pred_labels = pred_labels.reshape(-1, 1)
        if pred_labels.shape[1] == 1 and F > 1:
            pred_labels = np.tile(pred_labels, (1, F))

    lossTfinal = normalize_score(np.mean(lossT, axis=1))
    lossFinal  = normalize_score(np.mean(loss, axis=1))
    labelsFinal = (np.sum(labels, axis=1) >= 1).astype(int)
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
    result.update(hit_att(loss, labels))
    result.update(ndcg(loss, labels))
    print(df)
    pprint(result)
    lossTfinal = normalize_score(np.mean(lossT, axis=1))
    lossFinal  = normalize_score(np.mean(loss,  axis=1))
    labelsFinal = (np.sum(labels, axis=1) >= 1).astype(int)
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
    result.update(hit_att(loss, labels))
    result.update(ndcg(loss, labels))
    print(f"\n==== beta_recon={beta_recon:.2f} (disc={1-beta_recon:.2f}) ====")
    pprint(result)
    row = {"beta_recon": beta_recon}
    row.update(result)
    results_beta.append(row)
df_beta = pd.DataFrame(results_beta)
df_beta.to_csv(f"beta_sweep_{args.model}_{args.dataset}.csv", index=False)
print(f"\nSaved: beta_sweep_{args.model}_{args.dataset}.csv")