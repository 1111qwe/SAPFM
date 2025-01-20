import argparse

def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    # ['ERA5', 'IMERG', 'GLDAS', 'GsMap', 'CMORPH', 'SP', 'T2M', 'RH', 'DEM'] 
    parser.add_argument('--input_dim', type=int, default=8)
    parser.add_argument('--sequence', type=int, default=3)
    parser.add_argument('--size', type=int, default=7)
    parser.add_argument('--version', type=str, default='contrast_convLSTM')
    
    # Training setting
    parser.add_argument('--nepoch', type=int, default=10000, help='how many times to update the generator')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=20)
    
    # using pretrained
    parser.add_argument('--pretrained_model', type=str, default='1.pth')

    parser.add_argument('--save_log', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=True)
    
    # Path
    parser.add_argument('--train_path', type=str, default='./data/era5_imerg_gldas_gsmap_cmorph_4additioans_lat_lon/2004_2014_t_1_s_3.npy')
    parser.add_argument('--test_path', type=str, default='./data/era5_imerg_gldas_gsmap_cmorph_4additioans_lat_lon/2015_2020_t_1_s_3.npy')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--pred_path', type=str, default='./predicts')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--model_save_name', type=str, default='3.pth')
    
    # box_cox
    parser.add_argument('--epsilon', type=float, default=1e-3)


    return parser.parse_args(args=[])

