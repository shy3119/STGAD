import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=False,
					default='synthetic',
                    help="dataset SMD")
parser.add_argument('--model', 
					metavar='-m', 
					type=str, 
					required=False,
					default='LSTM_Multivariate',
                    help="model name")
parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
parser.add_argument('--retrain', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--less', 
					action='store_true', 
					help="train using less data")
parser.add_argument('--explain', action='store_true', help="Run attention explainability visualization")
parser.add_argument('--noise_train', action='store_true', help='noise')
parser.add_argument("--plot", action="store_true",
                    help="Run plotting on test outputs (no retraining needed)")
parser.add_argument("--lsweep", type=str, default="", help='comma-separated window sizes, e.g. "5,10,20"')
parser.add_argument('--gan_lr', type=float, default=1e-4, help='learning rate for GAN optimizers (G and D)')
parser.add_argument('--match_n_train', type=int, default=5, help='N for best-of-N matching loss during training (0 disables)')
parser.add_argument('--match_agg_train', type=str, default='min', choices=['min','mean'], help='aggregation for matching loss')
parser.add_argument('--match_weight', type=float, default=1.0, help='weight for matching loss added to adversarial loss')

parser.add_argument('--stability', action='store_true', help='run stability evidence experiment (export step-level logs)')
parser.add_argument('--stability_models', type=str, default='MAD_GAN_Standard,MAD_GAN_WGANGP',
                    help='comma-separated model names for stability comparison')
parser.add_argument('--stability_seeds', type=str, default='1,2,3,4,5',
                    help='comma-separated random seeds')
parser.add_argument('--stability_epochs', type=int, default=5, help='epochs to train per run for stability logging')
parser.add_argument('--n_critic', type=int, default=5, help='number of critic updates per G update (WGAN-GP)')
parser.add_argument('--lambda_gp', type=float, default=5.0, help='gradient penalty weight (WGAN-GP)')
parser.add_argument('--stability_outdir', type=str, default='results/stability', help='output directory for stability logs')
parser.add_argument('--stability_burnin_frac', type=float, default=0.2, help='fraction of initial steps to ignore in summary stats')
parser.add_argument('--benchmark', action='store_true', help='Benchmark GPU throughput/latency/memory')
args = parser.parse_args()