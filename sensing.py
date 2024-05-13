import torch

def matched_filtering(signal_grad):
    list_flattened_grad = []
    for m in range(len(signal_grad)):
        flat_tensor = []
        for _, v in signal_grad[m].items():
            for tensor in v:
                flat_tensor.append(tensor.flatten())
        #print(flat_tensor)
        final_vector = torch.cat(flat_tensor)
        list_flattened_grad.append(final_vector)

    
    return [0] * len(signal_grad)