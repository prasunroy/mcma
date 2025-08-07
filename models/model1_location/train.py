import os, time, torch
from data import create_dataloader
from model import LocationPredictor


# configurations
# -----------------------------------------------------------------------------
embeds_dim = 128
latent_dim = 32
enable_gpu = True
learn_rate = 0.001
adam_beta1 = 0.5
adam_beta2 = 0.999
data_rootd = '../../datasets/binge_watching'
image_size = 256
batch_size = 8
num_epochs = 100
output_dir = './output'
# -----------------------------------------------------------------------------


# create model
model = LocationPredictor(embeds_dim, latent_dim)
if enable_gpu and torch.cuda.is_available():
    model = model.cuda()
num_params = 0
for param in model.parameters():
    num_params += param.numel()
print(f'[INFO] created model with {num_params/1e6:.1f}M parameters')


# create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, betas=(adam_beta1, adam_beta2))
print('[INFO] created optimizer')


# create dataloaders
train_dataloader = create_dataloader(data_rootd, 'train', image_size, batch_size, shuffle=True)
test_dataloader = create_dataloader(data_rootd, 'test', image_size, batch_size, shuffle=False)
print(f'[INFO] created dataloaders with {len(train_dataloader.dataset)} train',
      f'and {len(test_dataloader.dataset)} test samples')


# get inputs
def get_inputs(data):
    img = data['img']
    seg = data['seg']
    roi = data['scaled_roi_center']
    if enable_gpu and torch.cuda.is_available():
        img, seg, roi = img.cuda(), seg.cuda(), roi.cuda()
    return img, seg, roi


if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

num_batches = len(train_dataloader)
wid_batch = len(str(num_batches))
wid_epoch = len(str(num_epochs))
best_loss = float('inf')
losslimit = 40000

# training loop
for epoch in range(num_epochs):
    # optimization
    for i, batch in enumerate(train_dataloader):
        t0 = time.time()
        img, seg, roi = get_inputs(batch)
        p, mu, logvar = model(img, seg, roi)
        loss_mse = torch.nn.functional.mse_loss(p, roi)
        loss_kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        loss = loss_mse + loss_kld
        if loss < losslimit:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_suffix = 'OK'
        else:
            update_suffix = '-> update skipped'
        t1 = time.time()
        print(f'[TRAIN] epoch: {epoch+1:{wid_epoch}d}/{num_epochs} | batch: {i+1:{wid_batch}d}/{num_batches} |',
              f'mse: {loss_mse:10.4f} | kld: {loss_kld:10.4f} | time: {round(t1-t0, 2):.2f} sec | {update_suffix}')
    
    # evaluation
    print(f'[EVAL] evaluating model at the end of epoch {epoch+1}... ', end='')
    mode = model.training
    model.eval()
    loss = 0
    for i, batch in enumerate(test_dataloader):
        img, seg, roi = get_inputs(batch)
        with torch.no_grad():
            p, mu, logvar = model(img, seg, roi)
        loss_mse = torch.nn.functional.mse_loss(p, roi)
        loss_kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        loss += ((loss_mse.item() + loss_kld.item()) * p.size(0))
    loss /= len(test_dataloader)
    if best_loss is None or loss < best_loss:
        print(f'loss improved from {best_loss:.4f} to {loss:.4f}')
        best_loss = loss
        name_suffix = '_improved'
    else:
        print(f'loss did not improve from {best_loss:.4f}')
        name_suffix = ''
    save_path = os.path.join(output_dir, f'model_epoch_{str(epoch+1).zfill(wid_epoch)}{name_suffix}.pth')
    torch.save(model.state_dict(), save_path)
    print(f'[INFO] checkpoint saved to {save_path}')
    model.train(mode)
