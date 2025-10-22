# Import the necessary libraries
import torch
import torch.nn as nn
from torchinfo import summary


class HFT(nn.Module):
    """
        HFT implements the hFT-Transformer model for music transcription.
        It consists of an encoder and a decoder, where the encoder processes the input spectrogram
        and the decoder generates the MIDI outputs in two hierarchies: frequency and time.

        Args:
            encoder (HFTEncoder): The encoder part of the hFT-Transformer model.
            decoder (HFTDecoder): The decoder part of the hFT-Transformer model.
    """
    def __init__(self, encoder, decoder):
        """
            Initializes the HFT class.
        """
        super().__init__()
        self.hft_encoder = encoder
        self.hft_decoder = decoder
    
    def forward(self, spectrogram):
        """
            Forward pass for the HFT model.

            Args:
                spectrogram (torch.Tensor): The input spectrogram of shape (batch_size, margin_b+n_frame+margin_f, n_bin).

            Returns:
                out_on_1st, out_off_1st, out_frame_1st, out_velocity_1st: Outputs from the first hierarchy.
                attn_freq: Attention weights from the frequency dimension.
                out_on_2nd, out_off_2nd, out_frame_2nd, out_velocity_2nd: Outputs from the second hierarchy.
        """
        enc_output = self.hft_encoder(spectrogram)
        out_on_1st, out_off_1st, out_frame_1st, out_velocity_1st, \
        attn_freq, out_on_2nd, out_off_2nd, out_frame_2nd, out_velocity_2nd = self.hft_decoder(enc_output)
        return out_on_1st, out_off_1st, out_frame_1st, out_velocity_1st, \
               attn_freq, out_on_2nd, out_off_2nd, out_frame_2nd, out_velocity_2nd


class HFTEncoder(nn.Module):
    def __init__(self, n_margin, n_frame, n_bin, cnn_channel, cnn_kernel, d, n_layers, num_heads, pff_dim, dropout, device):
        """
            Initializes the HFTEncoder class.

            Args:
                n_margin (int): The margin size for the spectrogram.
                n_frame (int): The number of frames in the spectrogram.
                n_bin (int): The number of frequency bins in the spectrogram.
                cnn_channel (int): The number of channels in the CNN layer.
                cnn_kernel (int): The kernel size for the CNN layer.
                d (int): The dimension of the model.
                n_layers (int): The number of layers in the transformer encoder.
                num_heads (int): The number of attention heads.
                pff_dim (int): The dimension of the position-wise feed-forward network.
                dropout (float): Dropout rate.
                device (torch.device): The device to run the model on.
        """
        super().__init__()
        self.device = device
        self.n_frame = n_frame
        self.n_bin = n_bin
        self.cnn_channel = cnn_channel
        self.cnn_kernel = cnn_kernel
        self.d = d
        self.conv = nn.Conv2d(1, cnn_channel, kernel_size=(1, cnn_kernel))
        self.n_proc = n_margin * 2 + 1
        self.cnn_dim = self.cnn_channel * (self.n_proc - (self.cnn_kernel - 1))
        self.tok_embedding_freq = nn.Linear(self.cnn_dim, d)
        self.pos_embedding_freq = nn.Embedding(n_bin, d) 
        self.layers_freq = nn.ModuleList([
            TransformerEncoderLayer(d, num_heads, pff_dim, dropout, device) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.scale_freq = torch.FloatTensor([self.d ** 0.5]).to(device)
    
    def forward(self, spectrogram):
        """
            Forward pass for the HFTEncoder.

            Args:
                spectrogram (torch.Tensor): The input spectrogram of shape (batch_size, margin_b+n_frame+margin_f, n_bin). 

            Returns:
                enc_output (torch.Tensor): The encoded output of shape (batch_size, n_frame, n_bin, d).
        """
        batch_size = spectrogram.size(0)

        # spectrogram is of shape (batch_size, margin_b+n_frame+margin_f, n_bin)
        # however, we will do something interesting: we will take windows of size n_frame
        # and slide them across the spectrogram with a stride of 1.
        # (batch_size, n_frame, n_bin, n_proc)
        spectrogram = spectrogram.unfold(2, self.n_proc, 1).permute(0, 2, 1, 3).contiguous()  
        spec_cnn = spectrogram.reshape(batch_size*self.n_frame, self.n_bin, self.n_proc).unsqueeze(1)

        # (batch_size*n_frame, n_bin, cnn_channel, n_proc-(cnn_kernel-1))
        spec_cnn = self.conv(spec_cnn).permute(0, 2, 1, 3).contiguous()
        spec_cnn_freq = spec_cnn.reshape(batch_size*self.n_frame, self.n_bin, self.cnn_dim)

        # Get the embedding
        spec_emb_freq = self.tok_embedding_freq(spec_cnn_freq)  # (batch_size*n_frame, n_bin, d)

        # position encoding
        pos_freq = torch.arange(0, self.n_bin).unsqueeze(0).repeat(batch_size*self.n_frame, 1).to(self.device)
        spec_freq = self.dropout(spec_emb_freq * self.scale_freq + self.pos_embedding_freq(pos_freq))  # (batch_size*n_frame, n_bin, d)

        # Pass through the transformer encoder layers
        for layer in self.layers_freq:
            # spec_freq is of shape (batch_size*n_frame, n_bin, d)
            spec_freq = layer(spec_freq)
        
        # spec_freq is now of shape (batch_size*n_frame, n_bin, d)
        # Reshape it back to (batch_size, n_frame, n_bin, d)
        enc_output = spec_freq.reshape(batch_size, self.n_frame, self.n_bin, self.d)
        return enc_output  # (batch_size, n_frame, n_bin, d)
        

class HFTDecoder(nn.Module):
    """    
        HFTDecoder implements the decoder part of the hFT-Transformer model.
        This involves the two decoder sections (the one in the first hierarchy 
        and the one in the second hierarchy).
    """
    def __init__(self, n_frame, n_bin, n_note, n_velocity, d, n_layers, num_heads, pff_dim, dropout, device):
        """
            Initializes the HFTDecoder class.

            Args:
                n_frame (int): The number of frames in the spectrogram.
                n_bin (int): The number of frequency bins in the spectrogram.
                n_note (int): The number of MIDI notes.
                n_velocity (int): The number of MIDI velocities.
                d (int): The dimension of the model.
                n_layers (int): The number of layers in the decoder.
                num_heads (int): The number of attention heads.
                pff_dim (int): The dimension of the position-wise feed-forward network.
                dropout (float): Dropout rate.
                device (torch.device): The device to run the model on.
        """
        super().__init__()
        self.device = device
        self.n_frame = n_frame
        self.n_note = n_note
        self.n_velocity = n_velocity
        self.n_bin = n_bin
        self.d = d
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

        self.pos_embedding_freq = nn.Embedding(n_note, d)
        self.layer_zero_freq = TransformerDecoderLayerZero(d, num_heads, pff_dim, dropout, device)
        self.layers_freq = nn.ModuleList([
            TransformerDecoderLayer(d, num_heads, pff_dim, dropout, device) for _ in range(n_layers-1)
        ])

        self.fc_onset_freq = nn.Linear(d, 1)
        self.fc_offset_freq = nn.Linear(d, 1)
        self.fc_frame_freq = nn.Linear(d, 1)
        self.fc_velocity_freq = nn.Linear(d, n_velocity)

        self.scale_time = torch.FloatTensor([n_frame ** 0.5]).to(device)
        self.pos_embedding_time = nn.Embedding(n_frame, d)
        self.layers_time = nn.ModuleList([
            TransformerEncoderLayer(d, num_heads, pff_dim, dropout, device) for _ in range(n_layers)
        ])

        self.fc_onset_time = nn.Linear(d, 1)
        self.fc_offset_time = nn.Linear(d, 1)
        self.fc_frame_time = nn.Linear(d, 1)
        self.fc_velocity_time = nn.Linear(d, n_velocity)
    
    def forward(self, enc_output):
        """
            Forward pass for the HFTDecoder.    

            Args:
                enc_output (torch.Tensor): The encoded output from the HFTEncoder of shape (batch_size, n_frame, n_bin, d).
            
            Returns:
                out_on_1st, out_off_1st, out_frame_1st, out_velocity_1st: Outputs from the first hierarchy.
                attn_freq: Attention weights from the frequency dimension.
                out_on_2nd, out_off_2nd, out_frame_2nd, out_velocity_2nd: Outputs from the second hierarchy.    
        """
        batch_size = enc_output.size(0)
        enc_output = enc_output.reshape([batch_size*self.n_frame, self.n_bin, self.d])

        pos_freq = torch.arange(0, self.n_note).unsqueeze(0).repeat(batch_size*self.n_frame, 1).to(self.device)
        midi_freq = self.pos_embedding_freq(pos_freq)  # (batch_size*n_frame, n_note, d)

        midi_freq, attn_freq = self.layer_zero_freq(enc_output, midi_freq)
        for layer in self.layers_freq:
            # midi_freq is of shape (batch_size*n_frame, n_note, d)
            # attn_freq is of shape (batch_size*n_frame, num_heads, n_note d)
            midi_freq, attn_freq = layer(enc_output, midi_freq)
        
        # attn_freq is now of shape (batch_size, n_frame, num_heads, n_note, d)
        attn_freq = attn_freq.reshape(batch_size, self.n_frame, attn_freq.shape[1], attn_freq.shape[2], attn_freq.shape[3]) 

        # output_1st hierarchy (freq dimension)
        out_on_1st = self.sigmoid(self.fc_onset_freq(midi_freq).reshape(batch_size, self.n_frame, self.n_note))  # (batch_size, n_frame, n_note)
        out_off_1st = self.sigmoid(self.fc_offset_freq(midi_freq).reshape(batch_size, self.n_frame, self.n_note))  # (batch_size, n_frame, n_note)
        out_frame_1st = self.sigmoid(self.fc_frame_freq(midi_freq).reshape(batch_size, self.n_frame, self.n_note)) # (batch_size, n_frame, n_note)
        
        # batch_size, n_frame, n_note, n_velocity for out_velocity_1st
        out_velocity_1st = self.fc_velocity_freq(midi_freq).reshape(batch_size, self.n_frame, self.n_note, self.n_velocity)

        ## Second Hierarchy (Time Dimension)
        midi_time = midi_freq.reshape([batch_size, self.n_frame, self.n_note, \
        self.d]).permute(0, 2, 1, 3).contiguous().reshape([batch_size*self.n_note, \
                                self.n_frame, self.d])  # (batch_size*n_note, n_frame, d)
        pos_time = torch.arange(0, self.n_frame).unsqueeze(0).repeat(batch_size*self.n_note, 1).to(self.device)
        midi_time = self.dropout(midi_time*self.scale_time + self.pos_embedding_time(pos_time))  # (batch_size*n_note, n_frame, d)

        for layer in self.layers_time:
            # midi_time is of shape (batch_size*n_note, n_frame, d)
            midi_time = layer(midi_time)

        # output_2nd hierarchy (time dimension)
        out_on_2nd = self.sigmoid(self.fc_onset_time(midi_time).reshape(batch_size, self.n_note, self.n_frame).permute(0, 2, 1).contiguous())  # (batch_size, n_frame, n_note)
        out_off_2nd = self.sigmoid(self.fc_offset_time(midi_time).reshape(batch_size, self.n_note, self.n_frame).permute(0, 2, 1).contiguous())
        out_frame_2nd = self.sigmoid(self.fc_frame_time(midi_time).reshape(batch_size, self.n_note, self.n_frame).permute(0, 2, 1).contiguous())
        out_velocity_2nd = self.fc_velocity_time(midi_time).reshape(batch_size, self.n_note, self.n_frame, self.n_velocity).permute(0, 2, 1, 3).contiguous()
        return out_on_1st, out_off_1st, out_frame_1st, out_velocity_1st, \
               attn_freq, out_on_2nd, out_off_2nd, out_frame_2nd, out_velocity_2nd


class TransformerDecoderLayerZero(nn.Module):
    def __init__(self, d, num_heads, pff_dim, dropout, device):
        """
            Initializes the TransformerDecoderLayerZero class.

            Args:
                d (int): The dimension of the input.
                num_heads (int): The number of attention heads.
                pff_dim (int): The dimension of the position-wise feed-forward network.
                dropout (float): Dropout rate.
                device (torch.device): The device to run the model on.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(d)
        self.cross_attn = MultiHeadAttention(d, num_heads, dropout, device)
        self.ff = FeedForward(d, pff_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, enc_output, dec_input):
        """
            Forward pass for the TransformerDecoderLayerZero.
            
            Args:
                enc_output (torch.Tensor): The encoded output from the HFTEncoder of shape (batch_size, n_frame, n_bin, d).
                dec_input (torch.Tensor): The decoder input of shape (batch_size, n_note, d).   

            Returns:
                dec_input (torch.Tensor): The updated decoder input after self-attention and cross-attention.
                cross_attn_weights (torch.Tensor): The attention weights from the cross-attention layer.
        """
        # cross-attention with encoder output
        attn_enc, cross_attn_weights = self.cross_attn(dec_input, enc_output, enc_output)  # Cross-attention with encoder output
        dec_input = self.layer_norm(dec_input + self.dropout(attn_enc))  # Residual connection and layer normalization
        
        # Feed-forward network
        # dec_input is of shape (batch_size, seq_len, d)
        ff = self.ff(dec_input)
        dec_input = self.layer_norm(dec_input + self.dropout(ff))  # Residual connection
        
        # (batch_size, seq_len, d), cross_attn_weights of shape (batch_size, num_heads, seq_len_dec, seq_len_enc)
        return dec_input, cross_attn_weights  

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d, num_heads, pff_dim, dropout, device):
        """
            Initializes the TransformerDecoderLayer class.

            Args:
                d (int): The dimension of the input.
                num_heads (int): The number of attention heads.
                pff_dim (int): The dimension of the position-wise feed-forward network.
                dropout (float): Dropout rate.
                device (torch.device): The device to run the model on.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(d)
        self.self_attn = MultiHeadAttention(d, num_heads, dropout, device)
        self.cross_attn = MultiHeadAttention(d, num_heads, dropout, device)
        self.ff = FeedForward(d, pff_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, enc_output, dec_input):
        """
            Forward pass for the TransformerDecoderLayer.

            Args:
                enc_output (torch.Tensor): The encoded output from the HFTEncoder of shape (batch_size, n_frame, n_bin, d).
                dec_input (torch.Tensor): The decoder input of shape (batch_size, seq_len_dec, d).

            Returns:
                dec_input (torch.Tensor): The updated decoder input after self-attention and cross-attention.
                cross_attn_weights (torch.Tensor): The attention weights from the cross-attention layer.
        """
        attn_dec, _ = self.self_attn(dec_input, dec_input, dec_input)  # Self-attention on decoder input
        dec_input = self.layer_norm(dec_input + self.dropout(attn_dec))  # Residual connection and layer normalization

        # cross-attention with encoder output
        attn_enc, cross_attn_weights = self.cross_attn(dec_input, enc_output, enc_output)  # Cross-attention with encoder output
        dec_input = self.layer_norm(dec_input + self.dropout(attn_enc))  # Residual connection and layer normalization
        
        # Feed-forward network
        # dec_input is of shape (batch_size, seq_len, d)
        ff = self.ff(dec_input)
        dec_input = self.layer_norm(dec_input + self.dropout(ff))  # Residual connection

        # (batch_size, seq_len, d), cross_attn_weights of shape (batch_size, num_heads, seq_len_dec, seq_len_enc)
        return dec_input, cross_attn_weights  


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d, num_heads, pff_dim, dropout, device):
        """
        TransformerEncoderLayer implements a single layer of the Transformer encoder.
        It consists of multi-head self-attention and position-wise feed-forward networks.

        Args:
            d (int): The dimension of the input.
            num_heads (int): The number of attention heads.
            pff_dim (int): The dimension of the position-wise feed-forward network.
            dropout (float): Dropout rate.
            device (torch.device): The device to run the model on.
    """
        super().__init__()
        self.layer_norm = nn.LayerNorm(d)
        self.self_attn = MultiHeadAttention(d, num_heads, dropout, device)
        self.ff = FeedForward(d, pff_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
            Forward pass for the TransformerEncoderLayer.

            Args:
                x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d).

            Returns:    
                x (torch.Tensor): The output tensor after applying self-attention and feed-forward network.
        """
        # x is of shape (batch_size, seq_len, d)
        # pass model through self-attention
        attn_out, _ = self.self_attn(x, x, x)  # (batch_size, seq_len, d)
        x = self.layer_norm(x + attn_out)  # Residual connection and layer normalization
        ff = self.ff(x)  # (batch_size, seq_len, d)
        x = self.layer_norm(x + self.dropout(ff))  # Residual connection and layernorm
        return x  # (batch_size, seq_len, d)

class MultiHeadAttention(nn.Module):
    def __init__(self, d, num_heads, dropout, device):
        """
            Initializes the MultiHeadAttention class.

            Args:
                d (int): The dimension of the input.
                num_heads (int): The number of attention heads.
                dropout (float): Dropout rate.
                device (torch.device): The device to run the model on.
        """

        super().__init__()
        assert d % num_heads == 0, "Dimension must be divisible by number of heads."
        self.d = d
        self.num_heads = num_heads
        self.dh = d // num_heads # Dimension of each head
        self.fc_query = nn.Linear(d, d)
        self.fc_key = nn.Linear(d, d)
        self.fc_value = nn.Linear(d, d)
        self.fc_out = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.FloatTensor([self.dh ** 0.5]).to(device)

    def forward(self, query, key, value):
        """
            Forward pass for the MultiHeadAttention layer.

            Args:
                query (torch.Tensor): The query tensor of shape (batch_size, seq_len_q, d).
                key (torch.Tensor): The key tensor of shape (batch_size, seq_len_k, d).
                value (torch.Tensor): The value tensor of shape (batch_size, seq_len_v, d).

            Returns:
                output (torch.Tensor): The output tensor after applying multi-head attention.
                attn_weights (torch.Tensor): The attention weights.
        """

        # query, key and value are of shape (batch_size, seq_len, d)
        batch_size = query.size(0)

        # Linear projections
        q = self.fc_query(query)
        k = self.fc_key(key)
        v = self.fc_value(value)

        # Reshape to (batch_size, seq_len, num_heads, dh)
        q = q.view(batch_size, -1, self.num_heads, self.dh).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.num_heads, self.dh).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.num_heads, self.dh).permute(0, 2, 1, 3)

        # scaled dot-product attention
        similarity = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale # (batch_size, num_heads, seq_len_q, seq_len_k)
        attn_weights = torch.softmax(similarity, dim=-1)


        self_attn = torch.matmul(self.dropout(attn_weights), v)

        # (batch_size, seq_len_q, num_heads, dh)
        self_attn = self_attn.permute(0, 2, 1, 3).contiguous() 
        self_attn = self_attn.view(batch_size, -1, self.d)  # (batch_size, seq_len_q, d)
        output = self.fc_out(self_attn)  # (batch_size, seq_len_q, d)
        return output, attn_weights

class FeedForward(nn.Module):
    def __init__(self, hidden_dim, pff_dim, dropout):
        """
            Initializes the FeedForward class.

            Args:
                hidden_dim (int): The dimension of the hidden layer.
                pff_dim (int): position feedforward dimension.
                dropout (float): Dropout rate.
        """
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, pff_dim)
        self.fc2 = nn.Linear(pff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
            Forward pass for the FeedForward layer.

            Args:
                x (torch.Tensor): The input tensor of shape (batch_size, seq_len, hidden_dim).

            Returns:
                x (torch.Tensor): The output tensor after applying position embedding.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

decoder = HFTDecoder(
    n_frame=128,
    n_bin=256,  # Number of frequency bins  
    n_note=88,  # Number of MIDI notes
    n_velocity=128,  # Number of MIDI velocities
    d=256,  # Dimension of the model
    n_layers=3,  # Number of layers in the decoder
    num_heads=4,  # Number of attention heads
    pff_dim=512,  # Dimension of the position-wise feed-forward network
    dropout=0.1,  # Dropout rate
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #
    # Device to run the model on
)

encoder = HFTEncoder(
    n_margin=32,  # Margin size for the spectrogram
    n_frame=128,  # Number of frames in the spectrogram
    n_bin=256,  # Number of frequency bins in the spectrogram
    cnn_channel=4,  # Number of channels in the CNN layer
    cnn_kernel=5,  # Kernel size for the CNN layer
    d=256,  # Dimension of the model        
    n_layers=3,  # Number of layers in the transformer encoder
    num_heads=4,  # Number of attention heads
    pff_dim=512,  # Dimension of the position-wise feed-forward network
    dropout=0.1,  # Dropout rate
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #
    # Device to run the model on
)

if __name__ == "__main__":
    """
        Main function to create an instance of the HFT model and print the number of trainable parameters.
    """
    model = HFT(encoder, decoder)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters in the model: {count_parameters(model)}")
    # encoder number of trainable parameters
    print(f"Number of trainable parameters in the encoder: {count_parameters(encoder)}")
    # decoder number of trainable parameters
    print(f"Number of trainable parameters in the decoder: {count_parameters(decoder)}")

    batch_size = 4  # Example batch size
    num_frames = 128  # Example number of frames
    num_notes = 88
    summary(model, input_size=(batch_size, 256, 192))
