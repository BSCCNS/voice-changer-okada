import torch
from torch import device
from voice_changer.RVC.embedder.Embedder import Embedder
from transformers import HubertModel, HubertConfig
import os


class ApplioContentvec(Embedder):
    def loadModel(self, model_dir: str, dev: device, isHalf: bool = True) -> Embedder:
        """
        Load Applio's ContentVec model from a Hugging Face-style directory.
        Expected files:
            - config.json
            - pytorch_model.bin
        """
        super().setProps("applio-contentvec", model_dir, dev, isHalf)

        print(f"[ApplioContentVec] Loading model from: {model_dir}")
        config = HubertConfig.from_pretrained(model_dir)
        model = HubertModel.from_pretrained(model_dir, config=config)

        model = model.to(dev)
        model.eval()
        if isHalf:
            model = model.half()

        self.model = model
        return self

    def extractFeatures(
        self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True
    ) -> torch.Tensor:
        """
        Extract features from the Applio ContentVec model.
        - `embOutputLayer` is ignored because Hugging Face HubertModel doesn't expose intermediate layers directly.
        - `useFinalProj` is also ignored because this model lacks final_proj.
        """
        # Construct dummy attention mask (no padding)
        attention_mask = torch.ones(feats.shape[:-1], dtype=torch.long).to(self.dev)

        with torch.no_grad():
            outputs = self.model(
                feats.to(self.dev),
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            # If embOutputLayer is valid (e.g., 9), use hidden state from that layer
            if embOutputLayer < len(outputs.hidden_states):
                feats_out = outputs.hidden_states[embOutputLayer]
            else:
                feats_out = outputs.last_hidden_state  # fallback

        return feats_out
