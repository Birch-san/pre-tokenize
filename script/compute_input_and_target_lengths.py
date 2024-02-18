import argparse
from pre_tokenize.transformers.compute_input_and_target_lengths import compute_input_and_target_lengths

def main():
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  p.add_argument('--input-length', type=int, default=512)
  p.add_argument('--mlm-probability', type=float, default=.15)
  p.add_argument('--mean-noise-span-length', type=float, default=3.)
  args = p.parse_args()
  before_mask_input_length, target_length = compute_input_and_target_lengths(
    inputs_length=args.input_length,
    noise_density=args.mlm_probability,
    mean_noise_span_length=args.mean_noise_span_length,
  )
  print(f'before_mask_input_length: {before_mask_input_length}\ntarget_length: {target_length}')

if __name__ == '__main__':
  main()