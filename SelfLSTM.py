import torch.nn as nn
import torch

class SelfAttention(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, bias):
        super(SelfAttention, self).__init__()
        self.Q = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=kernel_size, bias=bias)
        self.K = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=kernel_size, bias=bias)
        self.V = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=kernel_size, bias=bias)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.output = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1)

    def forward(self, x):
        bacth, C, H, W= x.shape
        Q = self.Q(x).reshape(bacth, -1, H*W)
        K = self.K(x).reshape(bacth, -1, H*W)
        V = self.V(x).reshape(bacth, C, H*W)
    
        attention = self.softmax(torch.matmul(Q.permute(0,2,1),K))
        selfAtten = torch.matmul(V, attention.permute(0,2,1))
        selfAtten = selfAtten.reshape(bacth, C, H, W)
        
        selfAtten = self.gamma*selfAtten + x
        out = self.output(selfAtten)
        return out

class SelfLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):

        super(SelfLSTMCell, self).__init__()
        assert kernel_size[0] == 1
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.conv = SelfAttention(input_dim=self.input_dim + self.hidden_dim,
                              output_dim=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              bias=self.bias)


    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.Q.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.Q.weight.device))


class SelfLSTM(nn.Module):  
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=1,
                 batch_first=False, bias=True, bidirectional=False):
        super(SelfLSTM, self).__init__()
        
        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.bidirectional = bidirectional
        
        #convLSTM forward direction
        cell_fw = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_fw.append(SelfLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_fw = nn.ModuleList(cell_fw) 
        
        #ConvLSTM backward direction
        cell_bw = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_bw.append(SelfLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_bw = nn.ModuleList(cell_bw)
    
    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        layer_outputs of shape (b, num_layer, t, hidden_dim * num_directions, h, w) include the output featurs(h_t) from  the last layer of the ConLSTM, for each t.
        
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
            
        b, seq_len, _, h, w = input_tensor.size()
            
        
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state,hidden_state_inv  = self._init_hidden(batch_size=b,
                                             image_size=(h, w))
        
        # ConvLSTM forward direction       
        layer_output_fw = []
        last_state_fw = []

        input_fw = input_tensor

        for layer_idx in range(self.num_layers):

            h_fw, c_fw = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h_fw, c_fw = self.cell_fw[layer_idx](input_tensor=input_fw[:, t, :, :, :], cur_state=[h_fw, c_fw])
                output_inner.append(h_fw)

            layer_output = torch.stack(output_inner, dim=1)
            input_fw = layer_output

            layer_output_fw.append(layer_output)
            last_state_fw.append([h_fw, c_fw])

        layer_outputs = torch.stack(layer_output_fw, dim=1) #hidden states of all layers H
 
        #convLSTM inverse dirrection
        if self.bidirectional is True:
            layer_output_bw = []
            last_state_bw = []

            input_inv = input_tensor
            for layer_idx in range(self.num_layers):
                h_inv, c_inv = hidden_state_inv[layer_idx]
                output_inner_inv = []
                for t in range(seq_len-1, -1,-1):
                    h_inv, c_inv = self.cell_bw[layer_idx](input_tensor=input_inv[:, t, :, :, :], cur_state=[h_fw, c_inv])
                    output_inner_inv.append(h_inv)
                
                output_inner_inv.reverse()
                layer_output_inv = torch.stack(output_inner_inv, dim=1)
                input_inv = layer_output
                
                layer_output_bw.append(layer_output_inv)
                last_state_bw.append([h_inv, c_inv])
                
            layer_outputs = torch.stack([torch.cat((layer_output_fw[i], layer_output_bw[i]), dim=2)for i in range(self.num_layers)], dim=1)
        return layer_outputs[:, -1, :, :, :, :], (last_state_fw, last_state_bw) if self.bidirectional is True else (last_state_fw, )          
              
    def _init_hidden(self, batch_size, image_size):
        init_states_fw = []
        init_states_bw = None
         
        for i in range(self.num_layers):
            init_states_fw.append(self.cell_fw[i].init_hidden(batch_size, image_size))
        if self.bidirectional is True:
            init_states_bw = []
            for i in range(self.num_layers):
                init_states_bw.append(self.cell_bw[i].init_hidden(batch_size, image_size))              
        return init_states_fw, init_states_bw
            
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')
        
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    
if __name__ == "__main__":
    x = torch.rand((2, 11, 7, 3, 3)) #(b, t, c, h, w)
    model = SelfLSTM(input_dim=7,
                        hidden_dim = 5, 
                        kernel_size=(1,1),
                        num_layers = 1,
                        batch_first = True,bidirectional=True)
    all_states, _ = model(x)
    print(all_states.shape)