import numpy as np

class RandomSampler:
    # https://codereview.stackexchange.com/questions/223569/generating-latin-hypercube-samples-with-numpy

    @classmethod
    def _check_num_is_even(cls, num):
        if num % 2 != 0:
            raise ValueError("Number of samples must be even")

    @classmethod
    def get_lh_sample(cls, param_mins, param_maxes, num_samples):
        dim = param_mins.size

        latin_points = np.array([np.random.permutation(num_samples) for _ in range(dim)]).T

        lengths = (param_maxes - param_mins)[None, :]
        samples = lengths*(latin_points + 0.5)/num_samples + param_mins[None, :]
        return samples

    @classmethod
    def get_uniform_samples(cls, param_mins, param_maxes, num_samples):
        dim = param_mins.size
        samples = np.concatenate([np.random.uniform(param_mins[i], param_maxes[i], size=(num_samples, 1)) for i in range(dim)], axis=-1)
        return samples

    @classmethod
    def get_sym_sample(cls, param_mins, param_maxes, num_samples):
        cls._check_num_is_even(num_samples)

        dim = param_mins.size

        even_nums = np.arange(0, num_samples, 2)
        permutations = np.array([np.random.permutation(even_nums) for i in range(dim)])
        inverses = (num_samples - 1) - permutations

        latin_points = np.concatenate((permutations,inverses), axis=1).T

        lengths = (param_maxes - param_mins)[None, :]
        return  lengths*(latin_points + 0.5)/num_samples + param_mins[None, :]