uv pip install "ray[default]==2.55.1"
uv pip install "rlinf-openpi==0.1.1" --no-deps
uv pip install tqdm-loggable==0.2
uv pip install flax==0.10.2
uv pip install openpi-client
uv pip install beartype==0.19.0
uv pip install jaxtyping==0.2.36
uv pip install numpydantic==1.6.11
uv pip install transformers==4.53.2
uv pip install augmax>=0.3.4
uv pip install ml_collections==1.0.0
cp -r /opt/venvs/python310_torch29_cuda/lib/python3.10/site-packages/openpi/models_pytorch/transformers_replace/* /opt/venvs/python310_torch29_cuda/lib/python3.10/site-packages/transformers/
uv pip install fsspec[gcs]>=2024.6.0