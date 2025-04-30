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

We need to look at the embedder of the RVC model, which is implemented by the class `FairseqHubert` inside

```
server/voice_changer/RVC/embedder/FairseqHubert.py
```

Just for testing, I’m printing the embedding, stored in logits defined in [line] 
(https://github.com/BSCCNS/voice-changer-okada/blob/14101c1bf54037add3b6c116c6d6bbda73068b60/server/voice_changer/RVC/embedder/FairseqHubert.py#L41
)

You will see this on the terminal, alongside with the timestamp which is useful to measure lag. 

Another possible point of extraction is on the `Pipeline.py` in `RVC` class (line 182)
```python
# embedding
feats = self.extractFeatures(feats, embOutputLayer, useFinalProj)
t.record("extract-feats")
```
however, a little later the features are projected if we are using an index file -- not sure which should we use, the raw features or the index features. In line 254 we have the final features used for audio generation, so maybe those!
** Remember** that these features are in the GPU so we must bring them to CPU and also reshape them so they fit a dataframe. I think the `feats_buffer` variable (line 257) does that...what I don't know is why it is returned! 


Now, this tensor needs to go through the projection with Umap, and then rendering with TouchDesigner or something similar
