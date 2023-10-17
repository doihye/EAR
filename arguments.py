import argparse
import numpy as np

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--domain_name')

	parser.add_argument('--task_name')

	parser.add_argument('--frame_stack', type=int)

	parser.add_argument('--action_repeat', type=int)

	parser.add_argument('--episode_length', type=int)

	parser.add_argument('--mode', default='train', type=str)

	parser.add_argument('--init_steps', type=int)

	parser.add_argument('--train_steps', type=int)

	parser.add_argument('--batch_size', type=int)

	parser.add_argument('--hidden_dim', type=int)

	parser.add_argument('--save_freq', type=int)

	parser.add_argument('--eval_freq', type=int)

	parser.add_argument('--eval_episodes', type=int)

	parser.add_argument('--critic_lr', type=float)

	parser.add_argument('--critic_beta', type=float)

	parser.add_argument('--critic_tau', type=float)

	parser.add_argument('--critic_target_update_freq', type=int)

	parser.add_argument('--actor_lr', type=float)

	parser.add_argument('--actor_beta', type=float)

	parser.add_argument('--actor_log_std_min', type=float)

	parser.add_argument('--actor_log_std_max', type=float)

	parser.add_argument('--actor_update_freq', type=int)

	parser.add_argument('--encoder_feature_dim', type=int)

	parser.add_argument('--encoder_lr', type=float)

	parser.add_argument('--encoder_tau', type=float)

	parser.add_argument('--use_ss', action='store_true')

	parser.add_argument('--ss_lr', type=float)

	parser.add_argument('--ss_update_freq', type=int)

	parser.add_argument('--num_layers', type=int)

	parser.add_argument('--num_shared_layers', type=int)

	parser.add_argument('--num_filters', type=int)

	parser.add_argument('--discount', type=float)

	parser.add_argument('--init_temperature', type=float)

	parser.add_argument('--alpha_lr', type=float)

	parser.add_argument('--alpha_beta', type=float)

	parser.add_argument('--seed', type=int)

	parser.add_argument('--work_dir', type=str)

	parser.add_argument('--save_model', default=True)

	parser.add_argument('--save_video', default=False)

	parser.add_argument('--ear_checkpoint', default=None, type=str)
	
	parser.add_argument('--ear_num_episodes', default=0, type=int)

	args = parser.parse_args()

	assert args.mode in {'train', 'color_easy', 'color_hard'} or 'video' in args.mode, f'unrecognized mode "{args.mode}"'
	assert args.seed is not None, 'must provide seed for experiment'
	assert args.work_dir is not None, 'must provide a working directory for experiment'

	if args.ear_checkpoint is not None:
		try:
			args.ear_checkpoint = args.ear_checkpoint.replace('k', '000')
			args.ear_checkpoint = int(args.ear_checkpoint)
		except:
			return ValueError('ear_checkpoint must be int, received', args.ear_checkpoint)
	
	return args
