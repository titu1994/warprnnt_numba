# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import torch

from warprnnt_numba.rnnt_loss.rnnt_numpy import RNNTLoss as RNNTLoss_Numpy
from warprnnt_numba.rnnt_loss.rnnt_atomic_locks import RNNTLossNumbaAtomicLock
from warprnnt_numba import numba_utils
from warprnnt_numba.numba_utils import __NUMBA_MINIMUM_VERSION__

DEVICES = ['cpu']

if torch.cuda.is_available():
    DEVICES.append('cuda')


def wrap_and_call(fn, acts, labels, device):
    acts = torch.FloatTensor(acts)
    if 'cuda' in device:
        acts = acts.cuda()
    acts.requires_grad = True

    lengths = [acts.shape[1]] * acts.shape[0]
    label_lengths = [len(l) for l in labels]
    labels = torch.IntTensor(labels)
    lengths = torch.IntTensor(lengths)
    label_lengths = torch.IntTensor(label_lengths)
    if 'cuda' in device:
        labels = labels.cuda()
        lengths = lengths.cuda()
        label_lengths = label_lengths.cuda()

    costs = fn(acts, labels, lengths, label_lengths)
    cost = torch.sum(costs)
    cost.backward()

    if 'cuda' in device:
        torch.cuda.synchronize()

    return costs.data.cpu().numpy(), acts.grad.data.cpu().numpy()


class TestRNNTLossPytorchAtomicLocks:

    @pytest.mark.parametrize('device', DEVICES)
    def test_case_small(self, device):
        if device == 'cuda':
            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        acts = np.array(
            [
                [
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.2, 0.8, 0.1]],
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1, 0.1], [0.7, 0.1, 0.2, 0.1, 0.1]],
                ]
            ]
        )
        labels = [[1, 2]]

        fn_pt = RNNTLossNumbaAtomicLock(blank=0, reduction='sum')
        pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

        fn_np = RNNTLoss_Numpy()
        np_cost, np_grads = wrap_and_call(fn_np, acts, labels, device)

        expected_cost = 4.495666
        expected_grads = np.array(
            [
                [
                    [
                        [-0.13116688, -0.3999269, 0.17703125, 0.17703125, 0.17703125],
                        [-0.18572757, 0.12247056, -0.18168412, 0.12247056, 0.12247056],
                        [-0.32091254, 0.06269141, 0.06928472, 0.12624499, 0.06269141],
                    ],
                    [
                        [0.05456069, -0.21824276, 0.05456069, 0.05456069, 0.05456069],
                        [0.12073959, 0.12073959, -0.48295835, 0.12073959, 0.12073959],
                        [-0.6925882, 0.16871116, 0.18645467, 0.16871116, 0.16871116],
                    ],
                ]
            ]
        )

        assert np.allclose(pt_cost, expected_cost, rtol=1e-6), "small_test costs mismatch."
        assert np.allclose(pt_grads, expected_grads), "small_test gradient mismatch."

        assert np.allclose(pt_cost, np_cost, rtol=1e-6), "small_test costs mismatch."
        assert np.allclose(pt_grads, np_grads), "small_test gradient mismatch."

    @pytest.mark.parametrize('device', DEVICES)
    def test_case_small_random(self, device):
        if device == 'cuda':
            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        rng = np.random.RandomState(0)
        acts = rng.randn(1, 4, 3, 3)
        labels = [[1, 2]]

        fn_pt = RNNTLossNumbaAtomicLock(blank=0, reduction='sum')
        pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

        fn_np = RNNTLoss_Numpy()
        np_cost, np_grads = wrap_and_call(fn_np, acts, labels, device)

        assert np.allclose(pt_cost, np_cost, rtol=1e-6), "small_random_test costs mismatch."
        assert np.allclose(pt_grads, np_grads), "small_random_test gradient mismatch."

    @pytest.mark.parametrize('device', DEVICES)
    @pytest.mark.parametrize('fastemit_lambda', [1.0, 0.01, 0.00001])
    def test_case_small_random_fastemit_reg(self, device, fastemit_lambda):
        if device == 'cuda':
            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        rng = np.random.RandomState(0)
        acts = rng.randn(1, 4, 3, 3)
        labels = [[1, 2]]

        fn_pt = RNNTLossNumbaAtomicLock(blank=0, reduction='sum', fastemit_lambda=fastemit_lambda)
        pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

        fn_np = RNNTLoss_Numpy(fastemit_lambda=fastemit_lambda)
        np_cost, np_grads = wrap_and_call(fn_np, acts, labels, device)

        assert np.allclose(pt_cost, np_cost, rtol=1e-6), "small_random_test costs mismatch."
        assert np.allclose(pt_grads, np_grads, atol=1e-5, rtol=1e-5), "small_random_test gradient mismatch."

    @pytest.mark.parametrize('device', DEVICES)
    def test_case_big_tensor(self, device):
        if device == 'cuda':
            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        # minibatch x T x U x alphabet_size
        activations = [
            [
                [
                    [0.06535690384862791, 0.7875301411923206, 0.08159176605666074],
                    [0.5297155426466327, 0.7506749639230854, 0.7541348379087998],
                    [0.6097641124736383, 0.8681404965673826, 0.6225318186056529],
                ],
                [
                    [0.6685222872103057, 0.8580392805336061, 0.16453892311765583],
                    [0.989779515236694, 0.944298460961015, 0.6031678586829663],
                    [0.9467833543605416, 0.666202507295747, 0.28688179752461884],
                ],
                [
                    [0.09418426230195986, 0.3666735970751962, 0.736168049462793],
                    [0.1666804425271342, 0.7141542198635192, 0.3993997272216727],
                    [0.5359823524146038, 0.29182076440286386, 0.6126422611507932],
                ],
                [
                    [0.3242405528768486, 0.8007644367291621, 0.5241057606558068],
                    [0.779194617063042, 0.18331417220174862, 0.113745182072432],
                    [0.24022162381327106, 0.3394695622533106, 0.1341595066017014],
                ],
            ],
            [
                [
                    [0.5055615569388828, 0.051597282072282646, 0.6402903936686337],
                    [0.43073311517251, 0.8294731834714112, 0.1774668847323424],
                    [0.3207001991262245, 0.04288308912457006, 0.30280282975568984],
                ],
                [
                    [0.6751777088333762, 0.569537369330242, 0.5584738347504452],
                    [0.08313242153985256, 0.06016544344162322, 0.10795752845152584],
                    [0.7486153608562472, 0.943918041459349, 0.4863558118797222],
                ],
                [
                    [0.4181986264486809, 0.6524078485043804, 0.024242983423721887],
                    [0.13458171554507403, 0.3663418070512402, 0.2958297395361563],
                    [0.9236695822497084, 0.6899291482654177, 0.7418981733448822],
                ],
                [
                    [0.25000547599982104, 0.6034295486281007, 0.9872887878887768],
                    [0.5926057265215715, 0.8846724004467684, 0.5434495396894328],
                    [0.6607698886038497, 0.3771277082495921, 0.3580209022231813],
                ],
            ],
        ]

        expected_costs = [4.2806528590890736, 3.9384369822503591]
        expected_grads = [
            [
                [
                    [-1.86843902e-01, -6.25548810e-02, 2.49398798e-01],
                    [-2.03376666e-01, 2.02399328e-01, 9.77333169e-04],
                    [-1.41016081e-01, 7.91234672e-02, 6.18926100e-02],
                ],
                [
                    [-1.15517676e-02, -8.12802389e-02, 9.28319991e-02],
                    [-1.54257029e-01, 2.29432687e-01, -7.51756504e-02],
                    [-2.46593088e-01, 1.46404594e-01, 1.00188486e-01],
                ],
                [
                    [-1.29182907e-02, -6.15932420e-02, 7.45115355e-02],
                    [-5.59857301e-02, 2.19830811e-01, -1.63845062e-01],
                    [-4.97626871e-01, 2.09239945e-01, 2.88386941e-01],
                ],
                [
                    [1.36048580e-02, -3.02196294e-02, 1.66147724e-02],
                    [1.13924511e-01, 6.27811998e-02, -1.76705718e-01],
                    [-6.67078257e-01, 3.67658824e-01, 2.99419403e-01],
                ],
            ],
            [
                [
                    [-3.56343776e-01, -5.53474613e-02, 4.11691219e-01],
                    [-9.69219357e-02, 2.94591039e-02, 6.74628317e-02],
                    [-6.35175705e-02, 2.76544970e-02, 3.58630717e-02],
                ],
                [
                    [-1.54499024e-01, -7.39420280e-02, 2.28441030e-01],
                    [-1.66789949e-01, -8.78955179e-05, 1.66877866e-01],
                    [-1.72369644e-01, 1.05565332e-01, 6.68043196e-02],
                ],
                [
                    [2.38748826e-02, -1.18255816e-01, 9.43809375e-02],
                    [-1.04707085e-01, -1.08934477e-01, 2.13641584e-01],
                    [-3.69844258e-01, 1.80118099e-01, 1.89726159e-01],
                ],
                [
                    [2.57137045e-02, -7.94617534e-02, 5.37480488e-02],
                    [1.22328237e-01, -2.38788679e-01, 1.16460443e-01],
                    [-5.98686993e-01, 3.02203178e-01, 2.96483815e-01],
                ],
            ],
        ]

        activations = np.array(activations)
        labels = [[1, 2], [1, 1]]

        fn_pt = RNNTLossNumbaAtomicLock(blank=0, reduction='sum')
        pt_costs, pt_grads = wrap_and_call(fn_pt, activations, labels, device)

        fn_np = RNNTLoss_Numpy()
        np_costs, np_grads = wrap_and_call(fn_np, activations, labels, device)

        assert np.allclose(pt_costs, sum(expected_costs)), "big_test average costs mismatch."
        assert np.allclose(pt_grads, expected_grads, rtol=1e-3), "big_test grads for average cost mismatch."

        assert np.allclose(pt_costs, np_costs), "big_test average costs mismatch."
        assert np.allclose(pt_grads, np_grads, rtol=1e-3), "big_test grads for average cost mismatch."

    @pytest.mark.parametrize('device', DEVICES)
    def test_case_large_random(self, device):
        if device == 'cuda':
            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        rng = np.random.RandomState(0)
        acts = rng.randn(4, 8, 11, 5)
        labels = [
            [1, 2, 4, 3, 2, 2, 1, 1, 1, 1],
            [3, 2, 2, 3, 4, 1, 1, 1, 1, 1],
            [4, 4, 1, 2, 1, 3, 4, 3, 1, 2],
            [1, 1, 2, 1, 2, 3, 3, 1, 1, 1],
        ]

        fn_pt = RNNTLossNumbaAtomicLock(blank=0, reduction='sum')
        pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

        fn_np = RNNTLoss_Numpy()
        np_cost, np_grads = wrap_and_call(fn_np, acts, labels, device)

        assert np.allclose(pt_cost, np_cost, atol=1e-5, rtol=1e-3), "large_random_test costs mismatch."
        assert np.allclose(pt_grads, np_grads, atol=1e-5, rtol=1e-3), "large_random_test gradient mismatch."

    @pytest.mark.parametrize('device', DEVICES)
    def test_case_small_clamp(self, device):
        if device == 'cuda':
            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        GRAD_CLAMP = 0.1
        acts = np.array(
            [
                [
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.2, 0.8, 0.1]],
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1, 0.1], [0.7, 0.1, 0.2, 0.1, 0.1]],
                ]
            ]
        )
        labels = [[1, 2]]

        fn_pt = RNNTLossNumbaAtomicLock(blank=0, reduction='sum', clamp=GRAD_CLAMP)
        pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

        fn_np = RNNTLoss_Numpy(blank=0, clamp=GRAD_CLAMP)
        np_cost, np_grads = wrap_and_call(fn_np, acts, labels, device)

        expected_cost = 4.495666
        expected_grads = np.array(
            [
                [
                    [
                        [-0.1, -0.1, 0.1, 0.1, 0.1],
                        [-0.1, 0.1, -0.1, 0.1, 0.1],
                        [-0.1, 0.06269141, 0.06928472, 0.1, 0.06269141],
                    ],
                    [
                        [0.05456069, -0.1, 0.05456069, 0.05456069, 0.05456069],
                        [0.1, 0.1, -0.1, 0.1, 0.1],
                        [-0.1, 0.1, 0.1, 0.1, 0.1],
                    ],
                ]
            ]
        )

        assert np.allclose(pt_cost, expected_cost, rtol=1e-6), "small_test costs mismatch."
        assert np.allclose(pt_grads, expected_grads), "small_test gradient mismatch."

        assert np.allclose(pt_cost, np_cost, rtol=1e-6), "small_test costs mismatch."
        assert np.allclose(pt_grads, np_grads), "small_test gradient mismatch."

    @pytest.mark.parametrize('device', DEVICES)
    @pytest.mark.parametrize('fastemit_lambda', [1.0, 0.01, 0.00001])
    def test_case_small_fastemit_clamp(self, device, fastemit_lambda):
        if device == 'cuda':
            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        GRAD_CLAMP = 0.1
        acts = np.array(
            [
                [
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.2, 0.8, 0.1]],
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1, 0.1], [0.7, 0.1, 0.2, 0.1, 0.1]],
                ]
            ]
        )
        labels = [[1, 2]]

        fn_pt = RNNTLossNumbaAtomicLock(blank=0, reduction='sum', fastemit_lambda=fastemit_lambda, clamp=GRAD_CLAMP)
        pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

        fn_np = RNNTLoss_Numpy(blank=0, fastemit_lambda=fastemit_lambda, clamp=GRAD_CLAMP)
        np_cost, np_grads = wrap_and_call(fn_np, acts, labels, device)

        expected_cost = 4.495666
        expected_cost += expected_cost * fastemit_lambda

        assert np.allclose(pt_cost, expected_cost, rtol=1e-6), "small_test costs mismatch."
        assert np.allclose(pt_cost, np_cost, rtol=1e-6), "small_test costs mismatch."
        assert np.allclose(pt_grads, np_grads), "small_test gradient mismatch."


if __name__ == "__main__":
    pytest.main([__file__])
