import torch


def compute_fractions(comp_data_tensor, BP, PIONA, mw, device='cpu'):
    comp_data_tensor = comp_data_tensor.to(device)
    BP = BP.to(device)
    PIONA = PIONA.to(device)
    mw = mw.to(device)

    temp_fraction = comp_data_tensor * mw
    temp_comp_data = temp_fraction / torch.sum(temp_fraction, dim=1, keepdim=True)

    gas_mask = (BP > 0) & (BP < 220)
    lpg_mask = (BP > 220) & (BP < 280)
    gasoline_mask = (BP > 280) & (BP < 473.15)
    diesel_mask = (BP > 473.15) & (BP < 623.15)
    coke_mask = ~ (gas_mask | lpg_mask | gasoline_mask | diesel_mask)

    gas_mask_tensor = gas_mask.float().to(device)
    lpg_mask_tensor = lpg_mask.float().to(device)
    gasoline_mask_tensor = gasoline_mask.float().to(device)
    diesel_mask_tensor = diesel_mask.float().to(device)
    coke_mask_tensor = coke_mask.float().to(device)

    gas_data = torch.sum(temp_comp_data * gas_mask_tensor, dim=1)
    lpg_data = torch.sum(temp_comp_data * lpg_mask_tensor, dim=1)
    gasoline_data = torch.sum(temp_comp_data * gasoline_mask_tensor, dim=1)
    diesel_data = torch.sum(temp_comp_data * diesel_mask_tensor, dim=1)
    coke_data = 1 - (gas_data + lpg_data + gasoline_data + diesel_data)

    p_mask_tensor = (PIONA == 1).float().to(device)
    i_mask_tensor = (PIONA == 2).float().to(device)
    o_mask_tensor = (PIONA == 3).float().to(device)
    n_mask_tensor = (PIONA == 4).float().to(device)
    a_mask_tensor = (PIONA == 5).float().to(device)

    p_data = torch.sum(temp_comp_data * p_mask_tensor * gasoline_mask_tensor, dim=1)
    i_data = torch.sum(temp_comp_data * i_mask_tensor * gasoline_mask_tensor, dim=1)
    o_data = torch.sum(temp_comp_data * o_mask_tensor * gasoline_mask_tensor, dim=1)
    n_data = torch.sum(temp_comp_data * n_mask_tensor * gasoline_mask_tensor, dim=1)
    a_data = torch.sum(temp_comp_data * a_mask_tensor * gasoline_mask_tensor, dim=1)

    mass_fraction = 100 * torch.stack(
        (gas_data, lpg_data, gasoline_data, diesel_data, coke_data), dim=1
    )

    mass_PIONA = 100 * torch.stack(
        (p_data, i_data, o_data, n_data, a_data), dim=1
    ) / torch.sum(
        torch.stack((p_data, i_data, o_data, n_data, a_data), dim=1), dim=1, keepdim=True
    )

    return mass_fraction, mass_PIONA


