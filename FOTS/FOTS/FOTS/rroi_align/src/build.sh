export TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__ -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__"

echo "Running script"
nvcc -c -o rroi_align -I /home/owner/.virtualenvs/fots/lib/python3.8/site-packages/torch/include -I /home/owner/.virtualenvs/fots/lib/python3.8/site-packages/torch/include/torch/csrc/api/include -I /home/owner/.virtualenvs/fots/lib/python3.8/site-packages/torch/include/torch/csrc/api/include/torch -I /home/owner/.virtualenvs/fots/lib/python3.8/site-packages/torch/include/torch/csrc/utils -I /home/owner/.virtualenvs/fots/lib/python3.8/site-packages/torch/include/TH -I /home/owner/.virtualenvs/fots/lib/python3.8/site-packages/torch/include/THC -I /usr/include/python3.8 rroi_align_kernel.cu ${TORCH_NVCC_FLAGS}   