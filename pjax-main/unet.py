from pjax import nn

class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, t_emb):
        x = self.conv1(x)
        x = self.relu(x)

        if t_emb is not None:
            time_emb = self.time_mlp(t_emb)
            x = x + time_emb[:, :, None, None]

        x = self.conv2(x)
        x = self.relu(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.block = SimpleBlock(in_channels, out_channels, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t_emb):
        x = self.block(x, t_emb)
        x_shortcut = x
        x = self.pool(x)
        return x, x_shortcut

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.block = SimpleBlock(in_channels + in_channels, out_channels, time_emb_dim)

    def forward(self, x, x_shortcut, t_emb):
        x = self.upsample(x)

        if x.shape[2:] != x_shortcut.shape[2:]:
            x = nn.functional.interpolate(x, size=x_shortcut.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, x_shortcut], dim=1)
        x = self.block(x, t_emb)
        return x

class Unet(nn.Module):
    '''
    Simplified Unet with basic CNNs and ReLU
    '''
    def __init__(self, timesteps, time_embedding_dim, in_channels=3, out_channels=2, base_dim=32, dim_mults=[2,4,8,16]):
        super().__init__()

        self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)

        dims = [base_dim] + [base_dim * m for m in dim_mults]
        channels = []
        for i in range(len(dims)-1):
            channels.append((dims[i], dims[i+1]))

        self.init_conv = nn.Conv2d(in_channels, base_dim, 3, padding=1)

        self.encoder_blocks = nn.ModuleList([EncoderBlock(c[0], c[1], time_embedding_dim) for c in channels])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(c[1], c[0], time_embedding_dim) for c in channels[::-1]])

        self.mid_block = SimpleBlock(channels[-1][1], channels[-1][1], time_embedding_dim)

        self.final_conv = nn.Conv2d(in_channels=base_dim, out_channels=out_channels, kernel_size=1)

    def forward(self, x, t=None):
        if t is not None:
            t_emb = self.time_embedding(t)
        else:
            t_emb = None

        x = self.init_conv(x)

        encoder_shortcuts = []
        for encoder_block in self.encoder_blocks:
            x, x_shortcut = encoder_block(x, t_emb)
            encoder_shortcuts.append(x_shortcut)

        x = self.mid_block(x, t_emb)

        encoder_shortcuts.reverse()
        for decoder_block, shortcut in zip(self.decoder_blocks, encoder_shortcuts):
            x = decoder_block(x, shortcut, t_emb)

        x = self.final_conv(x)

        return x