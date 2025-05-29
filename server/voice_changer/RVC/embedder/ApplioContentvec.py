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

        self.model = model.to("cpu")  # Force to CPU

        #model = model.to(dev)
        model.eval()
        if isHalf:
            model = model.half()

        self.model = model
        return self

    def extractFeatures(
        self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True
    ) -> torch.Tensor:
        # Ensure input is on CPU, and in float32
        feats = feats.detach().to("cpu").float()

        # Create attention mask (no padding), also on CPU
        attention_mask = torch.ones(feats.shape[:-1], dtype=torch.long, device="cpu")

        with torch.no_grad():
            outputs = self.model(
                feats,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            if embOutputLayer < len(outputs.hidden_states):
                feats_out = outputs.hidden_states[embOutputLayer]
            else:
                feats_out = outputs.last_hidden_state

        return feats_out
