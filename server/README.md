MMVC Server (plus a few hacks)

## Installing the environment

### Outisde of BSC

```
conda env create -f environment.yml
conda activate vcclient-dev-bsc
```

###  From BSC

We need to circumvent the default channel. I didn't find a clean way of doing this, so I am installing the environment with conda (and the necessary pacakges with conda-forge) and then the rest with pip. 

```
conda create -n vcclient-dev-bsc --override-channels -c conda-forge python=3.10 pip onnxruntime fairseq
conda activate vcclient-dev-bsc
pip install -r requirements.txt
```

(pip throws an error about fairseq but ignore it)

To run in CPU, which we will in this project

```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Finally, run the command to start the app

```
python3 MMVCServerSIO.py -p 18888 --https true \
    --content_vec_500 pretrain/checkpoint_best_legacy_500.pt  \
    --content_vec_500_onnx pretrain/content_vec_500.onnx \
    --content_vec_500_onnx_on true \
    --hubert_base pretrain/hubert_base.pt \
    --hubert_base_jp pretrain/rinna_hubert_base_jp.pt \
    --hubert_soft pretrain/hubert/hubert-soft-0d54a1f4.pt \
    --nsf_hifigan pretrain/nsf_hifigan/model \
    --crepe_onnx_full pretrain/crepe_onnx_full.onnx \
    --crepe_onnx_tiny pretrain/crepe_onnx_tiny.onnx \
    --rmvpe pretrain/rmvpe.pt \
    --model_dir server/model_dir \
    --samples samples.json
```

## Extracting latent space

To extract the embedding for the LS we go to the `Pipeline` class in `server/voice_changer/RVC/pipeline/Pipeline.py`
```python
# embedding
feats = self.extractFeatures(feats, embOutputLayer, useFinalProj)
t.record("extract-feats")
```

Then, the tensor `feats` goes through a 3D projection with Umap (we use a surrogate for low latency) and then transmitted by UDP. 
